import json
import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from ..engine import Engine


class _AsyncWriter:
    '''
    Background worker for async I/O operations.
    
    Runs in a separate daemon thread and processes tasks from a queue.
    '''

    def __init__(self, max_queue_size: int = 1000) -> None:
        '''
        Initializes the AsyncWriter.

        Args:
            max_queue_size (int): Maximum number of tasks in the queue.
        '''
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._pending_count: int = 0

    def start(self) -> None:
        '''Start the background worker thread.'''
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        '''Stop the background worker and wait for completion.'''
        with self._lock:
            if not self._running:
                return
            self._running = False
        
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def submit(self, func: Callable, *args, **kwargs) -> None:
        '''
        Submit a task to the queue.

        Args:
            func (Callable): Function to execute.
            *args: Arguments for the function.
            **kwargs: Keyword arguments for the function.
        '''
        if not self._running:
            func(*args, **kwargs)
            return
        
        with self._lock:
            self._pending_count += 1
        
        try:
            self._queue.put((func, args, kwargs), block=False)
        except queue.Full:
            with self._lock:
                self._pending_count -= 1
            func(*args, **kwargs)

    def flush(self, timeout: float = 30.0) -> None:
        '''
        Wait for all queued tasks to complete.

        Args:
            timeout (float): Maximum time to wait.
        '''
        sentinel = object()
        self._queue.put(sentinel)
        
        start_time = datetime.now()
        while self._pending_count > 0:
            if (datetime.now() - start_time).total_seconds() > timeout:
                break
            threading.Event().wait(0.01)

    def pending_count(self) -> int:
        '''Return the number of pending tasks.'''
        with self._lock:
            return self._pending_count

    def _worker_loop(self) -> None:
        '''Main loop for the background worker thread.'''
        while True:
            try:
                item = self._queue.get()
                
                if item is None:
                    break
                
                if isinstance(item, tuple) and len(item) == 3:
                    func, args, kwargs = item
                    try:
                        func(*args, **kwargs)
                    except Exception:
                        pass
                    
                    with self._lock:
                        self._pending_count -= 1
                
            except Exception:
                pass


class RecorderHub:
    '''
    Central hub for recording and persisting training artifacts.

    This plugin provides a unified interface for logging various types of data
    including scalars, text, images, audio, binary files, and documents.
    It also manages checkpoints and supports resuming from existing recordings.

    Attributes:
        enable (bool): Whether the recorder is active.
        root (Path): Root directory for all recordings (runs/<timestamp>_<name>/).
        experiment_name (str): Name of the experiment.
    '''

    def __init__(self, engine: 'Engine') -> None:
        '''
        Initializes the RecorderHub.

        Args:
            engine (Engine): The engine instance this recorder is attached to.
        '''
        self._engine = engine
        self._enable: bool = False
        self._root: Optional[Path] = None
        self._experiment_name: str = ''
        self._current_step: int = 0
        self._current_epoch: int = 0
        self._async_mode: bool = True
        self._writer: _AsyncWriter = _AsyncWriter()
        self._jsonl_lock: threading.Lock = threading.Lock()

    @property
    def enable(self) -> bool:
        '''Returns whether the recorder is enabled.'''
        return self._enable

    @property
    def root(self) -> Optional[Path]:
        '''Returns the root directory path.'''
        return self._root

    @property
    def experiment_name(self) -> str:
        '''Returns the experiment name.'''
        return self._experiment_name

    def set_recorder(
        self,
        path: str = '',
        name: str = '',
        resume: bool = False,
        async_mode: bool = True,
        queue_size: int = 1000,
        **kwargs
    ) -> 'RecorderHub':
        '''
        Initialize and enable the recorder.

        Args:
            path (str): Custom root path. If empty, uses default 'runs/' directory.
            name (str): Experiment name. If empty, uses timestamp only.
            resume (bool): Whether to resume from existing recording if path exists.
            async_mode (bool): Whether to use async I/O for logging.
            queue_size (int): Maximum queue size for async operations.
            **kwargs: Additional configuration options.

        Returns:
            RecorderHub: Self for method chaining.
        '''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f'{timestamp}_{name}' if name else timestamp

        if path:
            root_path = Path(path)
        else:
            root_path = Path('runs') / exp_name

        self._experiment_name = name if name else timestamp
        self._root = root_path
        self._async_mode = async_mode

        if resume and root_path.exists():
            self.resume_from(str(root_path))
        else:
            self._setup_directories()

        if self._async_mode:
            self._writer = _AsyncWriter(max_queue_size=queue_size)
            self._writer.start()

        self._enable = True

        self._engine.emit('recorder_ready', sender='recorder', data={
            'root': str(self._root),
            'async_mode': self._async_mode,
            'experiment_name': self._experiment_name,
        })

        return self

    def enable_recorder(self) -> None:
        '''Enable the recorder.'''
        self._enable = True

    def disable_recorder(self) -> None:
        '''Disable the recorder.'''
        self._enable = False

    def _setup_directories(self) -> None:
        '''Create the directory structure for recordings.'''
        if self._root is None:
            return

        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / 'texts').mkdir(exist_ok=True)
        (self._root / 'images').mkdir(exist_ok=True)
        (self._root / 'audio').mkdir(exist_ok=True)
        (self._root / 'binaries').mkdir(exist_ok=True)
        (self._root / 'documents').mkdir(exist_ok=True)
        (self._root / 'checkpoints').mkdir(exist_ok=True)

    def _get_step(self, step: Optional[int] = None) -> int:
        '''Get the current step, using engine step if not provided.'''
        if step is not None:
            return step
        return self._engine.step_global if hasattr(self._engine, 'step_global') else self._current_step

    def _get_epoch(self) -> int:
        '''Get the current epoch.'''
        return self._engine.epoch if hasattr(self._engine, 'epoch') else self._current_epoch

    def _write_jsonl(self, file_path: Path, data: Dict[str, Any]) -> None:
        '''Append a JSON line to a jsonl file (thread-safe).'''
        with self._jsonl_lock:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _write_jsonl_async(self, file_path: Path, data: Dict[str, Any]) -> None:
        '''Write to jsonl file (can be called from async thread).'''
        self._write_jsonl(file_path, data)

    def flush(self) -> None:
        '''Wait for all queued tasks to complete.'''
        pending = self._writer.pending_count()
        self._engine.emit('recorder_flush', sender='recorder', data={'pending': pending})
        self._writer.flush()

    def close(self) -> None:
        '''Stop the async writer and wait for completion.'''
        self._engine.emit('recorder_close', sender='recorder', data={})
        self._writer.stop()
        self._enable = False

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        '''
        Log a scalar metric.

        Args:
            tag (str): Tag name for the metric.
            value (float): The scalar value.
            step (Optional[int]): Step number. Uses engine step if not provided.
        '''
        if not self._enable or self._root is None:
            return

        current_step = self._get_step(step)
        current_epoch = self._get_epoch()

        self._engine.emit('log_scalar', sender='recorder', data={
            'tag': tag,
            'value': value,
            'step': current_step,
            'epoch': current_epoch,
        })

        record = {
            'tag': tag,
            'value': value,
            'step': current_step,
            'epoch': current_epoch,
            'timestamp': datetime.now().isoformat()
        }

        if self._async_mode:
            self._writer.submit(self._write_jsonl_async, self._root / 'metrics.jsonl', record)
        else:
            self._write_jsonl(self._root / 'metrics.jsonl', record)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        '''
        Log multiple scalar metrics under a single tag.

        Args:
            tag (str): Tag name for the metrics group.
            values (Dict[str, float]): Dictionary of metric names and values.
            step (Optional[int]): Step number. Uses engine step if not provided.
        '''
        if not self._enable or self._root is None:
            return

        current_step = self._get_step(step)
        current_epoch = self._get_epoch()

        self._engine.emit('log_scalars', sender='recorder', data={
            'tag': tag,
            'values': values,
            'step': current_step,
            'epoch': current_epoch,
        })

        for name, value in values.items():
            record = {
                'tag': f'{tag}/{name}',
                'value': value,
                'step': current_step,
                'epoch': current_epoch,
                'timestamp': datetime.now().isoformat()
            }
            if self._async_mode:
                self._writer.submit(self._write_jsonl_async, self._root / 'metrics.jsonl', record)
            else:
                self._write_jsonl(self._root / 'metrics.jsonl', record)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None
    ) -> None:
        '''
        Log text content.

        Args:
            tag (str): Tag name for the text.
            text (str): The text content to log.
            step (Optional[int]): Step number. Uses engine step if not provided.
        '''
        if not self._enable or self._root is None:
            return

        current_step = self._get_step(step)
        text_file = self._root / 'texts' / f'{tag}.txt'

        self._engine.emit('log_text', sender='recorder', data={
            'tag': tag,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'step': current_step,
            'path': str(text_file),
        })

        def _write_text():
            with open(text_file, 'a', encoding='utf-8') as f:
                f.write(f'=== Step {current_step} ===\n')
                f.write(text)
                f.write('\n\n')

            record = {
                'type': 'text',
                'tag': tag,
                'path': str(text_file),
                'step': current_step,
                'timestamp': datetime.now().isoformat()
            }
            self._write_jsonl(self._root / 'metrics.jsonl', record)

        if self._async_mode:
            self._writer.submit(_write_text)
        else:
            _write_text()

    def log_image(
        self,
        tag: str,
        image: Union[str, Path, 'torch.Tensor', Any],
        step: Optional[int] = None
    ) -> None:
        '''
        Log an image.

        Args:
            tag (str): Tag name for the image.
            image (Union[str, Path, torch.Tensor, np.ndarray]): Image data or path.
            step (Optional[int]): Step number. Uses engine step if not provided.
        '''
        if not self._enable or self._root is None:
            return

        current_step = self._get_step(step)
        image_dir = self._root / 'images' / tag
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f'step_{current_step}.png'

        self._engine.emit('log_image', sender='recorder', data={
            'tag': tag,
            'path': str(image_path),
            'step': current_step,
        })

        image_data = self._prepare_image_data(image)

        def _save_image():
            if image_data is not None:
                if isinstance(image_data, Path):
                    shutil.copy(image_data, image_path)
                else:
                    self._save_numpy_image(image_data, image_path)

            record = {
                'type': 'image',
                'tag': tag,
                'path': str(image_path),
                'step': current_step,
                'timestamp': datetime.now().isoformat()
            }
            self._write_jsonl(self._root / 'metrics.jsonl', record)

        if self._async_mode:
            self._writer.submit(_save_image)
        else:
            _save_image()

    def _prepare_image_data(self, image: Union[str, Path, 'torch.Tensor', Any]) -> Optional[Union[Path, Any]]:
        '''Prepare image data for saving (convert tensor to numpy).'''
        if isinstance(image, (str, Path)):
            src_path = Path(image)
            if src_path.exists():
                return src_path
            return None
        elif isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
        else:
            try:
                import numpy as np
                if isinstance(image, np.ndarray):
                    return image
            except ImportError:
                pass
            return None

    def _save_numpy_image(self, arr: Any, path: Path) -> None:
        '''Save a numpy array as an image.'''
        try:
            from PIL import Image
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != 'uint8':
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype('uint8')
            img = Image.fromarray(arr.squeeze())
            img.save(path)
        except ImportError:
            try:
                import torchvision.utils as vutils
                import torch
                tensor = torch.from_numpy(arr)
                if tensor.dim() == 4:
                    tensor = tensor[0]
                if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
                    vutils.save_image(tensor.float(), path)
            except ImportError:
                pass

    def log_audio(
        self,
        tag: str,
        audio: Union[str, Path, 'torch.Tensor', Any],
        sample_rate: int = 22050,
        step: Optional[int] = None
    ) -> None:
        '''
        Log audio data.

        Args:
            tag (str): Tag name for the audio.
            audio (Union[str, Path, torch.Tensor, np.ndarray]): Audio data or path.
            sample_rate (int): Sample rate for the audio.
            step (Optional[int]): Step number. Uses engine step if not provided.
        '''
        if not self._enable or self._root is None:
            return

        current_step = self._get_step(step)
        audio_dir = self._root / 'audio' / tag
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f'step_{current_step}.wav'

        self._engine.emit('log_audio', sender='recorder', data={
            'tag': tag,
            'path': str(audio_path),
            'sample_rate': sample_rate,
            'step': current_step,
        })

        audio_data = self._prepare_audio_data(audio)

        def _save_audio():
            if audio_data is not None:
                if isinstance(audio_data, Path):
                    shutil.copy(audio_data, audio_path)
                else:
                    self._save_numpy_audio(audio_data, audio_path, sample_rate)

            record = {
                'type': 'audio',
                'tag': tag,
                'path': str(audio_path),
                'sample_rate': sample_rate,
                'step': current_step,
                'timestamp': datetime.now().isoformat()
            }
            self._write_jsonl(self._root / 'metrics.jsonl', record)

        if self._async_mode:
            self._writer.submit(_save_audio)
        else:
            _save_audio()

    def _prepare_audio_data(self, audio: Union[str, Path, 'torch.Tensor', Any]) -> Optional[Union[Path, Any]]:
        '''Prepare audio data for saving (convert tensor to numpy).'''
        if isinstance(audio, (str, Path)):
            src_path = Path(audio)
            if src_path.exists():
                return src_path
            return None
        elif isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy()
        else:
            try:
                import numpy as np
                if isinstance(audio, np.ndarray):
                    return audio
            except ImportError:
                pass
            return None

    def _save_numpy_audio(self, arr: Any, path: Path, sample_rate: int) -> None:
        '''Save a numpy array as audio.'''
        try:
            import soundfile as sf
            sf.write(str(path), arr, sample_rate)
        except ImportError:
            try:
                import scipy.io.wavfile as wavfile
                import numpy as np
                arr = (arr * 32767).astype(np.int16)
                wavfile.write(str(path), sample_rate, arr)
            except ImportError:
                pass

    def log_binary(
        self,
        tag: str,
        data: bytes,
        filename: str,
        step: Optional[int] = None
    ) -> None:
        '''
        Log binary data.

        Args:
            tag (str): Tag name for the binary data.
            data (bytes): The binary data to save.
            filename (str): Filename for the binary file.
            step (Optional[int]): Step number. Uses engine step if not provided.
        '''
        if not self._enable or self._root is None:
            return

        current_step = self._get_step(step)
        binary_dir = self._root / 'binaries' / tag
        binary_dir.mkdir(parents=True, exist_ok=True)
        binary_path = binary_dir / filename

        self._engine.emit('log_binary', sender='recorder', data={
            'tag': tag,
            'path': str(binary_path),
            'filename': filename,
            'step': current_step,
            'size': len(data),
        })

        def _write_binary():
            with open(binary_path, 'wb') as f:
                f.write(data)

            record = {
                'type': 'binary',
                'tag': tag,
                'path': str(binary_path),
                'filename': filename,
                'step': current_step,
                'timestamp': datetime.now().isoformat()
            }
            self._write_jsonl(self._root / 'metrics.jsonl', record)

        if self._async_mode:
            self._writer.submit(_write_binary)
        else:
            _write_binary()

    def log_document(
        self,
        tag: str,
        content: Union[str, Dict[str, Any]],
        format: str = 'json'
    ) -> None:
        '''
        Log a document.

        Args:
            tag (str): Tag name for the document.
            content (Union[str, Dict[str, Any]]): Document content.
            format (str): Format of the document ('json' or 'text').
        '''
        if not self._enable or self._root is None:
            return

        doc_dir = self._root / 'documents'
        doc_path = doc_dir / f'{tag}.json' if format == 'json' else doc_dir / f'{tag}.txt'

        self._engine.emit('log_document', sender='recorder', data={
            'tag': tag,
            'path': str(doc_path),
            'format': format,
        })

        def _write_document():
            if format == 'json':
                with open(doc_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))

            record = {
                'type': 'document',
                'tag': tag,
                'path': str(doc_path),
                'format': format,
                'timestamp': datetime.now().isoformat()
            }
            self._write_jsonl(self._root / 'metrics.jsonl', record)

        if self._async_mode:
            self._writer.submit(_write_document)
        else:
            _write_document()

    def log_artifact(
        self,
        tag: str,
        file_path: Union[str, Path],
        copy: bool = True
    ) -> None:
        '''
        Log an external file as an artifact.

        Args:
            tag (str): Tag name for the artifact.
            file_path (Union[str, Path]): Path to the external file.
            copy (bool): Whether to copy the file. If False, only records the path.
        '''
        if not self._enable or self._root is None:
            return

        src_path = Path(file_path)
        if not src_path.exists():
            return

        artifact_dir = self._root / 'binaries' / tag
        artifact_dir.mkdir(parents=True, exist_ok=True)

        dest_path = artifact_dir / src_path.name if copy else src_path
        recorded_path = str(dest_path) if copy else str(src_path.absolute())

        self._engine.emit('log_artifact', sender='recorder', data={
            'tag': tag,
            'path': recorded_path,
            'copied': copy,
            'original_path': str(src_path),
        })

        def _copy_artifact():
            if copy:
                shutil.copy(src_path, dest_path)

            record = {
                'type': 'artifact',
                'tag': tag,
                'path': recorded_path,
                'copied': copy,
                'timestamp': datetime.now().isoformat()
            }
            self._write_jsonl(self._root / 'metrics.jsonl', record)

        if self._async_mode:
            self._writer.submit(_copy_artifact)
        else:
            _copy_artifact()

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        '''
        Save a checkpoint (synchronous operation).

        Args:
            state (Dict[str, Any]): State dictionary to save.
            filename (Optional[str]): Custom filename. If None, uses step number.

        Returns:
            Path: Path to the saved checkpoint.
        '''
        if self._root is None:
            raise RuntimeError('Recorder root is not set. Call set_recorder() first.')

        self.flush()

        checkpoint_dir = self._root / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            current_step = self._get_step()
            filename = f'checkpoint_step_{current_step}.pt'

        checkpoint_path = checkpoint_dir / filename
        torch.save(state, checkpoint_path)

        self._engine.emit('save_checkpoint', sender='recorder', data={
            'path': str(checkpoint_path),
            'step': self._get_step(),
        })

        record = {
            'type': 'checkpoint',
            'path': str(checkpoint_path),
            'step': self._get_step(),
            'timestamp': datetime.now().isoformat()
        }
        self._write_jsonl(self._root / 'metrics.jsonl', record)

        return checkpoint_path

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        '''
        Load a checkpoint (synchronous operation).

        Args:
            filename (str): Filename of the checkpoint to load.

        Returns:
            Dict[str, Any]: The loaded state dictionary.
        '''
        if self._root is None:
            raise RuntimeError('Recorder root is not set.')

        checkpoint_path = self._root / 'checkpoints' / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        state = torch.load(checkpoint_path, map_location='cpu')

        self._engine.emit('load_checkpoint', sender='recorder', data={
            'path': str(checkpoint_path),
        })

        return state

    def get_latest_checkpoint(self) -> Optional[Path]:
        '''
        Get the path to the latest checkpoint.

        Returns:
            Optional[Path]: Path to the latest checkpoint, or None if no checkpoints exist.
        '''
        if self._root is None:
            return None

        checkpoint_dir = self._root / 'checkpoints'
        if not checkpoint_dir.exists():
            return None

        checkpoints = list(checkpoint_dir.glob('*.pt'))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def resume_from(self, path: str) -> None:
        '''
        Resume from an existing recording directory.

        Args:
            path (str): Path to the existing recording directory.
        '''
        resume_path = Path(path)
        if not resume_path.exists():
            raise FileNotFoundError(f'Recording directory not found: {path}')

        self._root = resume_path
        self._enable = True

        config_path = resume_path / 'config.json'
        if config_path.exists():
            pass

        metrics_path = resume_path / 'metrics.jsonl'
        if metrics_path.exists():
            self._parse_existing_metrics(metrics_path)

        self._engine.emit('recorder_resume', sender='recorder', data={
            'path': str(resume_path),
        })

    def _parse_existing_metrics(self, metrics_path: Path) -> None:
        '''
        Parse existing metrics file to determine current progress.

        Args:
            metrics_path (Path): Path to the metrics.jsonl file.
        '''
        max_step = 0
        max_epoch = 0

        with open(metrics_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    step = record.get('step', 0)
                    epoch = record.get('epoch', 0)
                    max_step = max(max_step, step)
                    max_epoch = max(max_epoch, epoch)
                except json.JSONDecodeError:
                    continue

        self._current_step = max_step
        self._current_epoch = max_epoch

    def cleanup_future_records(
        self,
        current_step: int,
        current_epoch: int
    ) -> None:
        '''
        Remove records beyond the current step/epoch.

        This is useful when resuming training to remove records that
        were saved after the checkpoint.

        Args:
            current_step (int): Current step to keep records up to.
            current_epoch (int): Current epoch to keep records up to.
        '''
        if self._root is None:
            return

        self.flush()

        metrics_path = self._root / 'metrics.jsonl'
        if not metrics_path.exists():
            return

        kept_records = []
        with open(metrics_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    step = record.get('step', 0)
                    epoch = record.get('epoch', 0)

                    if step <= current_step and epoch <= current_epoch:
                        kept_records.append(line)
                except json.JSONDecodeError:
                    continue

        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.writelines(kept_records)

        self._cleanup_image_files(current_step)
        self._cleanup_audio_files(current_step)

        self._engine.emit('recorder_cleanup', sender='recorder', data={
            'current_step': current_step,
            'current_epoch': current_epoch,
        })

    def _cleanup_image_files(self, current_step: int) -> None:
        '''Remove image files beyond current step.'''
        if self._root is None:
            return

        images_dir = self._root / 'images'
        if not images_dir.exists():
            return

        for tag_dir in images_dir.iterdir():
            if not tag_dir.is_dir():
                continue
            for img_file in tag_dir.glob('step_*.png'):
                try:
                    step = int(img_file.stem.split('_')[1])
                    if step > current_step:
                        img_file.unlink()
                except (ValueError, IndexError):
                    continue

    def _cleanup_audio_files(self, current_step: int) -> None:
        '''Remove audio files beyond current step.'''
        if self._root is None:
            return

        audio_dir = self._root / 'audio'
        if not audio_dir.exists():
            return

        for tag_dir in audio_dir.iterdir():
            if not tag_dir.is_dir():
                continue
            for audio_file in tag_dir.glob('step_*.wav'):
                try:
                    step = int(audio_file.stem.split('_')[1])
                    if step > current_step:
                        audio_file.unlink()
                except (ValueError, IndexError):
                    continue

    def on_start_run(self, event: Any) -> None:
        '''Handle start_run event: dump config.json.'''
        if not self._enable or self._root is None:
            return

        config_path = self._root / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._engine.init_specs, f, ensure_ascii=False, indent=2)

        self._engine.emit('recorder_config_saved', sender='recorder', data={
            'path': str(config_path),
        })

    def on_start_epoch(self, event: Any) -> None:
        '''Handle start_epoch event.'''
        if not self._enable:
            return
        self._current_epoch = event.engine.epoch

    def on_end_epoch(self, event: Any) -> None:
        '''Handle end_epoch event.'''
        pass

    def on_before_step(self, event: Any) -> None:
        '''Handle before_step event.'''
        if not self._enable:
            return
        self._current_step = event.engine.step_global
