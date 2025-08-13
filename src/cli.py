"""
Command Line Interface for Master Generators
"""

import argparse
import sys
import json
import os
from typing import Dict, Any, Optional
import sympy as sp

from src.generators.master_generator import MasterGenerator
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
from src.ml.pattern_learner import GeneratorPatternLearner, create_model
from src.dl.novelty_detector import ODENoveltyDetector

class MasterGeneratorsCLI:
    """Command line interface for Master Generators"""
    
    def __init__(self):
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()
        self.linear_factory = LinearGeneratorFactory()
        self.nonlinear_factory = NonlinearGeneratorFactory()
        
    def generate_ode(self, args):
        """Generate a single ODE"""
        # Get the function
        if args.function_type == 'basic':
            f_z = self.basic_functions.get_function(args.function)
        else:
            f_z = self.special_functions.get_function(args.function)
        
        # Set parameters
        params = {
            'alpha': args.alpha,
            'beta': args.beta,
            'n': args.n,
            'M': args.M
        }
        
        # Generate ODE
        if args.type == 'linear':
            result = self.linear_factory.create(args.generator_number, f_z, **params)
        else:
            extra_params = {}
            if hasattr(args, 'q'):
                extra_params['q'] = args.q
            if hasattr(args, 'v'):
                extra_params['v'] = args.v
            if hasattr(args, 'a'):
                extra_params['a'] = args.a
            
            result = self.nonlinear_factory.create(
                args.generator_number, f_z, **{**params, **extra_params}
            )
        
        # Output results
        if args.output_format == 'json':
            # Convert sympy expressions to strings for JSON serialization
            output = {
                'ode': str(result['ode']),
                'solution': str(result['solution']),
                'type': result['type'],
                'order': result['order'],
                'generator_number': result['generator_number'],
                'initial_conditions': {k: str(v) for k, v in result['initial_conditions'].items()}
            }
            print(json.dumps(output, indent=2))
        elif args.output_format == 'latex':
            print(f"ODE: {sp.latex(result['ode'])}")
            print(f"Solution: {sp.latex(result['solution'])}")
        else:  # text
            print(f"ODE: {result['ode']}")
            print(f"Solution: {result['solution']}")
            print(f"Type: {result['type']}")
            print(f"Order: {result['order']}")
            print(f"Initial Conditions: {result['initial_conditions']}")
        
        # Save to file if requested
        if args.save:
            filename = f"ode_{args.type}_gen{args.generator_number}_{args.function}.json"
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved to {filename}")
    
    def batch_generate(self, args):
        """Generate multiple ODEs"""
        results = []
        
        for i in range(args.count):
            # Vary parameters slightly for each generation
            import numpy as np
            params = {
                'alpha': args.alpha + np.random.uniform(-0.5, 0.5),
                'beta': args.beta + np.random.uniform(-0.5, 0.5),
                'n': args.n,
                'M': args.M + np.random.uniform(-0.5, 0.5)
            }
            
            # Get random function
            if args.function_type == 'basic':
                func_names = self.basic_functions.get_function_names()
            else:
                func_names = self.special_functions.get_function_names()
            
            func_name = np.random.choice(func_names)
            
            if args.function_type == 'basic':
                f_z = self.basic_functions.get_function(func_name)
            else:
                f_z = self.special_functions.get_function(func_name)
            
            # Generate random generator number
            gen_num = np.random.randint(1, 9 if args.type == 'linear' else 11)
            
            # Generate ODE
            try:
                if args.type == 'linear':
                    result = self.linear_factory.create(gen_num, f_z, **params)
                else:
                    result = self.nonlinear_factory.create(gen_num, f_z, **params)
                
                results.append({
                    'id': i + 1,
                    'function': func_name,
                    'generator': gen_num,
                    'type': result['type'],
                    'order': result['order'],
                    'ode': str(result['ode'])[:100] + '...' if len(str(result['ode'])) > 100 else str(result['ode'])
                })
            except Exception as e:
                print(f"Error generating ODE {i+1}: {e}")
        
        # Output results
        if args.output_format == 'json':
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"ID: {r['id']}, Function: {r['function']}, Generator: {r['generator']}, Type: {r['type']}")
        
        # Save to file
        if args.save:
            filename = f"batch_odes_{args.type}_{args.count}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} ODEs to {filename}")
    
    def list_functions(self, args):
        """List available functions"""
        if args.function_type == 'basic' or args.function_type == 'all':
            print("\n=== Basic Functions ===")
            for name in sorted(self.basic_functions.get_function_names()):
                func = self.basic_functions.get_function(name)
                print(f"  {name}: {func}")
        
        if args.function_type == 'special' or args.function_type == 'all':
            print("\n=== Special Functions ===")
            for name in sorted(self.special_functions.get_function_names()):
                func = self.special_functions.get_function(name)
                print(f"  {name}: {func}")
    
    def analyze_novelty(self, args):
        """Analyze ODE novelty"""
        detector = ODENoveltyDetector()
        
        # Create a mock ODE dict
        ode_dict = {
            'ode': args.ode_expression,
            'type': args.type,
            'order': args.order
        }
        
        result = detector.check_novelty(ode_dict)
        
        print(f"Novelty Analysis Results:")
        print(f"  Novel: {'Yes' if result['is_novel'] else 'No'}")
        print(f"  Novelty Score: {result['novelty_score']}")
        print(f"  Complexity: {result['complexity_level']}")
        print(f"  Solvable by Standard Methods: {'Yes' if result['solvable_by_standard_methods'] else 'No'}")
        print(f"  Recommended Methods:")
        for method in result['recommended_methods']:
            print(f"    - {method}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Master Generators CLI - Generate and analyze ODEs'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate a single ODE')
    gen_parser.add_argument('--type', choices=['linear', 'nonlinear'], required=True)
    gen_parser.add_argument('--generator-number', type=int, required=True, help='Generator number')
    gen_parser.add_argument('--function', required=True, help='Function name')
    gen_parser.add_argument('--function-type', choices=['basic', 'special'], default='basic')
    gen_parser.add_argument('--alpha', type=float, default=1.0)
    gen_parser.add_argument('--beta', type=float, default=1.0)
    gen_parser.add_argument('--n', type=int, default=1)
    gen_parser.add_argument('--M', type=float, default=0.0)
    gen_parser.add_argument('--q', type=int, default=2, help='Power for nonlinear generators')
    gen_parser.add_argument('--v', type=int, default=3, help='Power for nonlinear generators')
    gen_parser.add_argument('--a', type=float, default=2.0, help='Scaling parameter')
    gen_parser.add_argument('--output-format', choices=['text', 'json', 'latex'], default='text')
    gen_parser.add_argument('--save', action='store_true', help='Save to file')
    
    # Batch generate command
    batch_parser = subparsers.add_parser('batch', help='Generate multiple ODEs')
    batch_parser.add_argument('--count', type=int, required=True, help='Number of ODEs to generate')
    batch_parser.add_argument('--type', choices=['linear', 'nonlinear', 'both'], default='both')
    batch_parser.add_argument('--function-type', choices=['basic', 'special', 'both'], default='basic')
    batch_parser.add_argument('--alpha', type=float, default=1.0)
    batch_parser.add_argument('--beta', type=float, default=1.0)
    batch_parser.add_argument('--n', type=int, default=1)
    batch_parser.add_argument('--M', type=float, default=0.0)
    batch_parser.add_argument('--output-format', choices=['text', 'json'], default='text')
    batch_parser.add_argument('--save', action='store_true', help='Save to file')
    
    # List functions command
    list_parser = subparsers.add_parser('list-functions', help='List available functions')
    list_parser.add_argument('--function-type', choices=['basic', 'special', 'all'], default='all')
    
    # Analyze novelty command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze ODE novelty')
    analyze_parser.add_argument('--ode-expression', required=True, help='ODE expression')
    analyze_parser.add_argument('--type', choices=['linear', 'nonlinear'], required=True)
    analyze_parser.add_argument('--order', type=int, required=True, help='Order of the ODE')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = MasterGeneratorsCLI()
    
    if args.command == 'generate':
        cli.generate_ode(args)
    elif args.command == 'batch':
        cli.batch_generate(args)
    elif args.command == 'list-functions':
        cli.list_functions(args)
    elif args.command == 'analyze':
        cli.analyze_novelty(args)
    elif args.command == 'version':
        from src import __version__
        print(f"Master Generators Version: {__version__}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()