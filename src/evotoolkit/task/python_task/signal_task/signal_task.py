import numpy as np
import torch
import types
from typing import Optional, Dict, Any, Callable
import sys
import warnings
import foolbox as fb
import evotoolkit
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import Solution, EvaluationResult
from evotoolkit.task.python_task.adversarial_attack.evo_attack import EvoAttack

# Setup device automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignalAttackTask(PythonTask):
    """
    Task definition for Black-box Adversarial Attacks on Raw I/Q Signals.
    
    This task adapts the L-AutoDA framework for modulation recognition models (e.g., VT-CNN2),
    optimizing perturbation algorithms to deceive the classifier with minimal L2 distortion.
    """
    
    def __init__(
        self, 
        model, 
        X_eval, 
        Y_eval, 
        attack_steps: int = 100,
        timeout_seconds: float = 300.0,
        use_mock: bool = False,
        starting_points = None
    ):
        self.model = model
        self.X_eval = X_eval
        self.Y_eval = Y_eval
        self.attack_steps = attack_steps
        self.use_mock = use_mock
        self.starting_points = starting_points
        
        # Initialize data for framework compatibility
        data = {
            "attack_steps": attack_steps,
            "n_test_samples": len(X_eval) if X_eval is not None else 0,
            "use_mock": use_mock
        }
        
        super().__init__(data=data, timeout_seconds=timeout_seconds)

    def _process_data(self, data: Dict[str, Any]):
        """
        Process input data and create task_info for logging.
        """
        self.data = data
        self.task_info = {
            "attack_steps": data["attack_steps"],
            "n_test_samples": data["n_test_samples"],
            "use_mock": data["use_mock"],
            "data_type": "radio_signal_raw_iq_2x128"  # Explicitly tag data type
        }

    def get_base_task_description(self) -> str:
        """
        Defines the system prompt for the LLM. 
        Specifies the context of RF Signal Adversarial Attacks.
        """
        return """You are an expert in Wireless Communication Security and Digital Signal Processing (DSP).

Task: Design an effective proposal generation algorithm for black-box adversarial attacks on RF signals.

Your goal is to evolve a `draw_proposals` function that generates high-quality candidate adversarial examples to fool a modulation classifier (VT-CNN2) while minimizing signal distortion (L2 norm).

**CRITICAL GOAL**: Your primary objective is to **reduce the L2 distance** (make the signal look like the original) while **keeping it adversarial** (maintaining the wrong classification).

Function Signature:
```python
def draw_proposals(
    org_sig: np.ndarray,
    best_adv_sig: np.ndarray,
    std_normal_noise: np.ndarray,
    hyperparams: np.ndarray
) -> np.ndarray:
    \"\"\"
    Generate a new candidate adversarial I/Q signal.

    Args:
        org_sig: Original clean I/Q signal, shape (2, 128), range [-1, 1].
                 Row 0 is In-phase (I), Row 1 is Quadrature (Q).
        best_adv_sig: Current best adversarial signal, shape (2, 128), range [-1, 1].
        std_normal_noise: Random normal noise, shape (2, 128).
        hyperparams: Evolutionary parameters to tune step sizes/angles.

    Returns:
        np.ndarray: New candidate adversarial signal, shape (2, 128).
    \"\"\"

Requirements:
- All inputs and outputs are numpy arrays
- Output must have same shape as org_sig: (2, 128)
- Output values should stay in [-1, 1] (will be clipped automatically)
- Use numpy operations (np.linalg.norm, np.dot, etc.)

Available Operations:
- Arithmetic: +, -, *, /
- Linear algebra: np.dot, np.linalg.norm, np.matmul
- Array operations: .reshape(), .flatten(), etc.

Strategy Guidelines:
1. **Direction**: Move from best_adv_sig towards the decision boundary (or towards org_sig to reduce distance)
2. **Step size**: Use hyperparams to control exploration vs exploitation
3. **Noise**: Incorporate std_normal_noise for exploration
4. **Distance**: Consider the vector from org_sig to best_adv_sig

Key Insights:
- The signal is not an image; it represents Phase and Amplitude.
- Smaller L2 distance from org_sig is better
- The candidate should be adversarial (fool the model)
- Balance between exploitation (refining best_adv_sig) and exploration (using noise)
- hyperparams adapts: increases when finding adversarials, decreases otherwise

Example Structure:
```python
import numpy as np

def draw_proposals(org_sig, best_adv_sig, std_normal_noise, hyperparams):
    
    # Reshape to vectors
    org = org_sig.flatten()
    best = best_adv_sig.flatten()
    noise = std_normal_noise.flatten()
    
    # Compute direction 
    diff = org - best

    # Your algorithm here: combine direction, noise, and hyperparams
    candidate = ...
    
    # Reshape back AND Clip to valid range [-1, 1]
    return np.clip(candidate.reshape(org_sig.shape), -1, 1)
```
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """
        Generates the initial baseline solution (Linear Interpolation + Noise).
        """
        initial_code = '''import numpy as np

def draw_proposals(org_sig, best_adv_sig, std_normal_noise, hyperparams):
    """
    Minimal Baseline: Linear interpolation towards original + Random Noise.
    """
    # 1. Direction vector towards original (Exploitation: reduce distance)
    diff = org_sig - best_adv_sig
    
    # 2. Step size controlled by hyperparams
    # Scale down for stability in signal space
    step = 0.01 * hyperparams[0]

    # 3. Combine: Move closer to original + Add exploration noise
    candidate = best_adv_sig + (step * diff) + (step * std_normal_noise)

    return candidate
'''
        print("Evaluating Initial Solution (Baseline)...")
        res = self.evaluate_code(initial_code)

        return Solution(sol_string=initial_code, evaluation_res=res, other_info={})

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """
        Executes and evaluates the code generated by LLM.
        """
        # Mock mode for quick debugging
        if self.use_mock:
            return EvaluationResult(
                valid=True,
                score=float(-np.random.uniform(0.1, 1.0)),
                additional_info={"avg_distance": float(np.random.uniform(0.1, 1.0)), "mock": True}
            )

        # Safe execution environment
        namespace = {
            "__builtins__": {
                "len": len, "range": range, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "sum": sum, "min": min, "max": max,
                "abs": abs, "print": print, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "tuple": tuple, "set": set,
                "__import__": __import__,
            },
            "np": np,
        }

        # 1. Execute code
        try:
            exec(candidate_code, namespace)
        except Exception as e:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": f"Code execution error: {str(e)}"})

        # 2. Check function existence
        if "draw_proposals" not in namespace:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "Function 'draw_proposals' not found"})

        draw_proposals = namespace["draw_proposals"]

        # 3. Run attack evaluation
        try:
            avg_dist = self._run_evo_attack(draw_proposals)
            
            if avg_dist is None or np.isnan(avg_dist) or np.isinf(avg_dist):
                 return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "Invalid distance returned"})

            # Score = -Distance (Maximize negative distance = Minimize distance)
            return EvaluationResult(
                valid=True, 
                score=-float(avg_dist), 
                additional_info={
                    "avg_distance": float(avg_dist),
                    "attack_steps": self.attack_steps
                }
            )
        except Exception as e:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": f"Attack evaluation error: {str(e)}"})
    
    def _run_evo_attack(self, draw_proposals_func: Callable) -> Optional[float]:
        """
        Runs the attack loop using evotoolkit's EvoAttack engine adapted for signals.
        """
        # 1. Wrap the function into a module for EvoAttack compatibility
        heuristic_module = types.ModuleType("heuristic_module")
        heuristic_module.draw_proposals = draw_proposals_func
        sys.modules[heuristic_module.__name__] = heuristic_module

        if torch.cuda.is_available():
            self.model.cuda()
          
        # Note: bounds=(-1, 1) matches the normalization of RML2016.10a
        fmodel = fb.PyTorchModel(self.model, bounds=(-1, 1), device=device)
        
        # Fallback initialization if starting_points are not provided
        if self.starting_points is not None:
            init_attack = None
        else:
            # This is slow, ideally starting_points are provided
            init_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(directions=1000, steps=1000)
        
        # 2. Initialize EvoAttack
        attack = EvoAttack(heuristic_module, init_attack=init_attack, steps=self.attack_steps)

        distances = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(len(self.X_eval)):
                # Ensure input is (1, 2, 128)
                x = self.X_eval[i:i+1].to(device)
                y = self.Y_eval[i:i+1].to(device)
                
                run_kwargs = {}
                if self.starting_points is not None:
                     if isinstance(self.starting_points, torch.Tensor):
                         sp = self.starting_points[i:i+1].to(device)
                     else:
                         sp = torch.tensor(self.starting_points[i:i+1]).to(device)
                     run_kwargs['starting_points'] = sp

                try:
                    # criteria=Misclassification means Untargeted Attack
                    img_adv = attack.run(fmodel, x, fb.criteria.Misclassification(y), **run_kwargs)

                    # Calculate L2 Norm on flattened signal (1, 256)
                    dist = torch.linalg.norm((x - img_adv).flatten(start_dim=1), axis=1).item()

                    if np.isnan(dist) or np.isinf(dist):
                        distances.append(10.0) # Penalty for failure
                    else:
                        distances.append(dist)

                except Exception as e:
                    # print(f"Sample {i} failed: {e}")
                    distances.append(10.0) # Penalty

        if not distances:
            return None

        return float(np.mean(distances))
