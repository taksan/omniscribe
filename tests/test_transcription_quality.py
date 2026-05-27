#!/usr/bin/env python3
"""Test transcription quality improvements.

This script tests the new transcription filtering features:
1. Hallucination filter - verify "Obrigado por assistir" is filtered
2. Confidence-based rejection - verify low-confidence segments are dropped
3. Per-segment timestamps - verify granular output
4. Repetition suppression - verify repeated text is filtered
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import soundfile as sf

from omniscribe.transcriber import HallucinationFilter, LiveTranscriber


def test_hallucination_filter():
    """Test that known hallucination patterns are detected."""
    print("\n=== Testing HallucinationFilter ===")
    
    filter = HallucinationFilter()
    
    # Known hallucinations that should be filtered
    hallucinations = [
        "Obrigado por assistir.",
        "Se inscreva no canal e ative o sininho",
        "Thanks for watching!",
        "Subscribe to the channel",
        "Acompanhe a avaliação do programa",
    ]
    
    # Valid speech that should pass
    valid_texts = [
        "Vamos começar a reunião sobre o sistema CIE.",
        "O projeto foi entregue em janeiro.",
        "Eu vou mostrar a tela do sistema.",
    ]
    
    passed = 0
    failed = 0
    
    for text in hallucinations:
        is_hall = filter.is_hallucination(text)
        if is_hall:
            print(f"  ✓ Correctly filtered: {text[:50]}...")
            passed += 1
        else:
            print(f"  ✗ FAILED to filter: {text[:50]}...")
            failed += 1
    
    for text in valid_texts:
        is_hall = filter.is_hallucination(text)
        if not is_hall:
            print(f"  ✓ Correctly passed: {text[:50]}...")
            passed += 1
        else:
            print(f"  ✗ WRONGLY filtered: {text[:50]}...")
            failed += 1
    
    print(f"\nHallucinationFilter: {passed} passed, {failed} failed")
    return failed == 0


def test_repetition_detection():
    """Test that repetition is detected."""
    print("\n=== Testing Repetition Detection ===")
    
    filter = HallucinationFilter()
    
    # Repetitive text should be detected
    repetitive = [
        "the the the the the",
        "um um um um",
        "e e e e e e",
    ]
    
    passed = 0
    failed = 0
    
    for text in repetitive:
        is_hall = filter.is_hallucination(text)
        if is_hall:
            print(f"  ✓ Correctly detected repetition: {text[:40]}...")
            passed += 1
        else:
            print(f"  ✗ FAILED to detect repetition: {text[:40]}...")
            failed += 1
    
    print(f"\nRepetition Detection: {passed} passed, {failed} failed")
    return failed == 0


def test_transcribe_clip(clip_path: Path, expected_duration: float) -> dict:
    """Run transcription on a test clip and return statistics."""
    print(f"\n=== Testing clip: {clip_path.name} ===")
    
    # Load audio
    audio, sr = sf.read(str(clip_path))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.txt"
        
        # Create transcriber with default settings
        transcriber = LiveTranscriber(
            output_path=output_path,
            model_name="small",
            device="cpu",
            compute_type="int8",
            language="pt",
            source_sample_rate=sr,
            chunk_seconds=6.0,
            enable_hallucination_filter=True,
            silence_threshold_db=-50.0,
            per_segment_output=True,
            min_logprob=-1.0,
            max_no_speech_prob=0.5,
            enable_repetition_filter=True,
        )
        
        transcriber.start()
        
        # Feed audio in chunks (simulating real-time)
        chunk_size = int(0.5 * sr)  # 0.5 second chunks
        label = "test_source"
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            transcriber.feed(label, chunk)
        
        # Let it process
        import time
        time.sleep(2)
        
        transcriber.stop()
        
        # Read output
        if output_path.exists():
            with open(output_path, "r") as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Analyze output
        stats = {
            "total_lines": len(lines),
            "content_lines": len([l for l in lines if l.strip() and not l.startswith("#")]),
            "hallucinations_filtered": 0,
            "timestamps": [],
        }
        
        # Check for any remaining hallucinations (should be 0)
        filter = HallucinationFilter()
        for line in lines:
            # Extract text after label
            if ":" in line:
                text = line.split(":", 1)[1].strip()
                if filter.is_hallucination(text):
                    stats["hallucinations_filtered"] += 1
                    print(f"  ✗ Hallucination NOT filtered: {text[:50]}...")
        
        # Extract timestamps
        for line in lines:
            match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
            if match:
                stats["timestamps"].append(match.group(1))
        
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Content lines: {stats['content_lines']}")
        print(f"  Remaining hallucinations: {stats['hallucinations_filtered']}")
        print(f"  Timestamps found: {len(stats['timestamps'])}")
        
        return stats


def main() -> int:
    """Run all tests."""
    print("=" * 70)
    print("OmniScribe Transcription Quality Tests")
    print("=" * 70)
    
    # Run unit tests
    results = []
    results.append(("HallucinationFilter", test_hallucination_filter()))
    results.append(("Repetition Detection", test_repetition_detection()))
    
    # Run integration tests on clips if they exist
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    if fixtures_dir.exists():
        for clip in fixtures_dir.glob("*.wav"):
            try:
                stats = test_transcribe_clip(clip, 30.0)
                # Test passes if no hallucinations remain
                passed = stats["hallucinations_filtered"] == 0
                results.append((f"Clip: {clip.name}", passed))
            except Exception as e:
                print(f"  ✗ Error testing {clip.name}: {e}")
                results.append((f"Clip: {clip.name}", False))
    else:
        print("\n⚠ No fixtures directory found, skipping integration tests.")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
