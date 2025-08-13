# tests/test_performance.py - Performance Tests
# ============================================================================
"""
Performance tests for critical components
"""

import pytest
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import memory_profiler
import cProfile
import pstats
from io import StringIO

from src.generators.master_generator import EnhancedMasterGenerator
from src.generators.linear_generators import LinearGeneratorFactory
from src.ml.pattern_learner import GeneratorPatternLearner
from src.dl.novelty_detector import ODENoveltyDetector

class TestPerformance:
    """Performance tests for system components"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance"""
        return EnhancedMasterGenerator(alpha=1.0, beta=1.0, n=1, M=0)
    
    @pytest.fixture
    def linear_factory(self):
        """Create linear factory instance"""
        return LinearGeneratorFactory()
    
    def test_generation_speed(self, generator):
        """Test ODE generation speed"""
        import sympy as sp
        f_z = sp.sin(sp.Symbol('z'))
        
        start_time = time.time()
        
        # Generate 100 ODEs
        for _ in range(100):
            y = generator.generate_y(f_z)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 10.0  # 10 seconds for 100 generations
        
        # Calculate average time per generation
        avg_time = elapsed_time / 100
        print(f"Average generation time: {avg_time:.4f} seconds")
    
    def test_parallel_generation(self, linear_factory):
        """Test parallel ODE generation"""
        import sympy as sp
        
        def generate_ode(i):
            f_z = sp.Symbol('z') ** (i % 3 + 1)
            return linear_factory.create((i % 8) + 1, f_z)
        
        # Sequential generation
        start_time = time.time()
        sequential_results = [generate_ode(i) for i in range(50)]
        sequential_time = time.time() - start_time
        
        # Parallel generation with threads
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(generate_ode, range(50)))
        parallel_time = time.time() - start_time
        
        # Parallel should be faster (or at least not much slower)
        print(f"Sequential time: {sequential_time:.4f}s")
        print(f"Parallel time: {parallel_time:.4f}s")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")
        
        # Results should be equivalent
        assert len(sequential_results) == len(parallel_results)
    
    def test_memory_usage(self, generator):
        """Test memory usage during generation"""
        import sympy as sp
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Generate many ODEs
        f_z = sp.exp(sp.Symbol('z'))
        results = []
        
        for i in range(100):
            y = generator.generate_y(f_z)
            results.append(str(y)[:100])  # Store truncated string
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to MB
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        print(f"Current memory: {current_mb:.2f} MB")
        print(f"Peak memory: {peak_mb:.2f} MB")
        
        # Memory usage should be reasonable
        assert peak_mb < 500  # Less than 500 MB for 100 generations
    
    def test_cache_performance(self):
        """Test cache performance improvement"""
        from src.utils.cache import CacheManager, cached
        
        cache = CacheManager(max_memory_size=100)
        
        # Function with expensive computation
        @cached(expire=3600)
        def expensive_function(n):
            time.sleep(0.01)  # Simulate expensive operation
            return n ** 2
        
        # First calls (no cache)
        start_time = time.time()
        for i in range(10):
            result = expensive_function(i)
        no_cache_time = time.time() - start_time
        
        # Second calls (with cache)
        start_time = time.time()
        for i in range(10):
            result = expensive_function(i)
        cache_time = time.time() - start_time
        
        # Cache should be significantly faster
        speedup = no_cache_time / cache_time
        print(f"No cache time: {no_cache_time:.4f}s")
        print(f"Cache time: {cache_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        assert speedup > 5  # At least 5x speedup with cache
    
    def test_ml_training_performance(self):
        """Test ML model training performance"""
        import torch
        from src.ml.trainer import MLTrainer
        
        trainer = MLTrainer(model_type='pattern_learner')
        
        start_time = time.time()
        
        # Train with small dataset
        trainer.train(
            epochs=5,
            batch_size=32,
            samples=100
        )
        
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f}s")
        
        # Should complete in reasonable time
        assert training_time < 60  # Less than 1 minute for small training
    
    def test_novelty_detection_speed(self):
        """Test novelty detection performance"""
        detector = ODENoveltyDetector()
        
        test_odes = [
            {"ode": "y''(x) + y(x) = 0", "type": "linear", "order": 2},
            {"ode": "(y''(x))^2 + y(x) = sin(x)", "type": "nonlinear", "order": 2},
            {"ode": "y'''(x) + y'(x) = exp(x)", "type": "linear", "order": 3},
        ]
        
        start_time = time.time()
        
        for ode_dict in test_odes * 10:  # Test 30 ODEs
            analysis = detector.check_novelty(ode_dict)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 30
        
        print(f"Average novelty detection time: {avg_time:.4f}s")
        
        # Should be fast
        assert avg_time < 0.5  # Less than 0.5 seconds per ODE
    
    def test_profiling(self, generator):
        """Profile code execution"""
        import sympy as sp
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run code to profile
        f_z = sp.sin(sp.Symbol('z'))
        for _ in range(10):
            y = generator.generate_y(f_z)
            y_prime = generator.generate_y_prime(f_z)
            y_double_prime = generator.generate_y_double_prime(f_z)
        
        profiler.disable()
        
        # Print profiling results
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        print("\nProfiling Results:")
        print(s.getvalue())
    
    @pytest.mark.benchmark
    def test_benchmark_generation(self, benchmark, generator):
        """Benchmark ODE generation"""
        import sympy as sp
        
        f_z = sp.exp(sp.Symbol('z'))
        
        # Benchmark the generation
        result = benchmark(generator.generate_y, f_z)
        
        assert result is not None
