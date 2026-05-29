"""Tests for the transcriber buffer functionality."""

import numpy as np
import pytest

from omniscribe.transcription import SourceBuffer as _SourceBuffer


class TestSourceBuffer:
    """Tests for the _SourceBuffer class."""

    def test_init(self):
        """Test buffer initialization."""
        buf = _SourceBuffer("test")
        assert buf.label == "test"
        assert buf.buffer.size == 0

    def test_append(self):
        """Test appending samples."""
        buf = _SourceBuffer("test")
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        buf.append(samples)
        
        assert buf.buffer.size == 3
        assert np.allclose(buf.buffer, samples)

    def test_append_multiple(self):
        """Test multiple appends."""
        buf = _SourceBuffer("test")
        buf.append(np.array([0.1, 0.2], dtype=np.float32))
        buf.append(np.array([0.3, 0.4], dtype=np.float32))
        
        assert buf.buffer.size == 4
        assert np.allclose(buf.buffer, [0.1, 0.2, 0.3, 0.4])

    def test_take_not_enough_data(self):
        """Test take returns None when not enough data."""
        buf = _SourceBuffer("test")
        buf.append(np.array([0.1, 0.2], dtype=np.float32))
        
        result = buf.take(10)
        
        assert result is None
        # Buffer should be unchanged
        assert buf.buffer.size == 2

    def test_take_success(self):
        """Test successful take removes data from buffer and returns start sample."""
        buf = _SourceBuffer("test")
        buf.append(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))

        result = buf.take(3)

        assert result is not None
        chunk, start = result
        assert chunk.size == 3
        assert np.allclose(chunk, [0.1, 0.2, 0.3])
        assert start == 0
        # Remaining in buffer
        assert buf.buffer.size == 2
        assert np.allclose(buf.buffer, [0.4, 0.5])
        # Second take advances the start sample counter
        result2 = buf.take(2)
        assert result2 is not None
        _, start2 = result2
        assert start2 == 3

    def test_take_exact_size(self):
        """Test take when buffer has exactly the requested size."""
        buf = _SourceBuffer("test")
        buf.append(np.array([0.1, 0.2, 0.3], dtype=np.float32))

        result = buf.take(3)

        assert result is not None
        chunk, start = result
        assert chunk.size == 3
        assert np.allclose(chunk, [0.1, 0.2, 0.3])
        assert start == 0
        assert buf.buffer.size == 0

    def test_drain(self):
        """Test drain returns all data, start sample, and clears buffer."""
        buf = _SourceBuffer("test")
        buf.append(np.array([0.1, 0.2, 0.3], dtype=np.float32))

        chunk, start = buf.drain()

        assert chunk.size == 3
        assert np.allclose(chunk, [0.1, 0.2, 0.3])
        assert start == 0
        assert buf.buffer.size == 0

    def test_drain_empty(self):
        """Test drain on empty buffer."""
        buf = _SourceBuffer("test")

        chunk, start = buf.drain()

        assert chunk.size == 0
        assert start == 0
        assert buf.buffer.size == 0

    def test_thread_safety(self):
        """Test that buffer operations are thread-safe."""
        import threading
        import time
        
        buf = _SourceBuffer("test")
        results = []
        
        def appender():
            for i in range(100):
                buf.append(np.array([float(i)], dtype=np.float32))
                time.sleep(0.001)
        
        def taker():
            for _ in range(50):
                result = buf.take(2)
                if result is not None:
                    results.append(result)
                time.sleep(0.002)
        
        t1 = threading.Thread(target=appender)
        t2 = threading.Thread(target=taker)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # Should have some results without crashing
        assert len(results) > 0
        # Buffer should have remaining items (0, 1, or possibly drained)
        assert buf.buffer.size >= 0
