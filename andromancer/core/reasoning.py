import json
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from andromancer.core.llm_client import AsyncLLMClient
from andromancer.core.memory import memory_store
from andromancer import config as cfg

logger = logging.getLogger("AndroMancer.Reasoning")

REACT_SYSTEM_PROMPT = """You are AndroMancer, an autonomous Android agent.

## Your Capabilities
{capabilities_json}

## Reasoning Protocol (ReAct)
For each step, think in this format:

**Thought**: Analyze current state and decide next actions. You can generate a sequence of actions (Macro-actions) to be executed in batch.
**Action Plan**: List of capabilities to execute.
**Confidence**: 0.0-1.0 how sure you are.

## Set-of-Mark (SoM) instructions
1. You will receive an image with numbered labels (e.g., [1], [2]).
2. These labels correspond to interactive elements on the screen.
3. When you want to interact with an element, you can refer to its label number in the `tap` or `type` capability using the `element` parameter (e.g., `{"element": {"label": 5}}`).
4. If you prefer coordinates, you can still use them, but labels are more reliable.

## Batching & Action Groups
1. Try to generate a sequence of steps in a single response if the flow is predictable (e.g., "Click login field, type user, click password field, type pass, click enter").
2. Only request more actions if the screen state is expected to change significantly or if validation is needed between steps.

## Output Format (JSON)
{{
    "reasoning": "step-by-step analysis of current situation",
    "action_plan": [
        {{
            "capability": "capability_name",
            "params": {{"param_key": "param_value"}},
            "expected_outcome": "what should happen",
            "critical": false,
            "async": false,
            "depends_on": null
        }}
    ],
    "confidence": 0.85,
    "next_steps": "what to do after this action succeeds"
}}

## Rules
1. ALWAYS analyze UI state before acting.
2. If `working_memory` contains `last_action_error`, prioritize fixing that error. If a `tap` failed due to missing coordinates, extract them from the UI summary and try again.
3. Use the coordinates (x, y) provided in the UI summary for `tap` actions. DO NOT emit a `tap` without `x` and `y` or a valid `element`.
4. If an action fails, analyze WHY and try an alternative approach.
5. Use memory of past experiences to avoid repeating mistakes.
6. For complex goals, decompose them into sub-tasks (e.g., 'Open WhatsApp', 'Send Message', 'Open YouTube').
7. If a non-critical sub-task fails (e.g., PiP mode not available), report it but CONTINUE with the next sub-task.
8. Use the `get_secret` capability to retrieve credentials when needed instead of asking the user immediately.
9. Use the `wait` capability if you expect a screen to take time to load.
10. NEVER execute high-risk actions without confirmation.
11. Consider suggestions from specialized skills if provided.
12. Use ONLY the parameters defined in the capability definition.

## Current Context
Goal: {goal}
Step: {step}/{max_steps}
Previous actions: {action_history}
Relevant memories: {memories}
{skill_context}
"""

@dataclass
class Thought:
    step: int
    reasoning: str
    action_plan: List[Dict]
    confidence: float
    observation: Optional[Dict] = None
    reflection: Optional[str] = None

class ReActEngine:
    """Reasoning + Acting implementation"""
    def __init__(self, llm_client: Optional[AsyncLLMClient] = None):
        self.llm = llm_client or AsyncLLMClient()
        self.thought_history: List[Thought] = []
        self.working_memory: Dict = {}

    async def reason(self, goal: str, observation: Dict, step: int, capabilities: List[Dict], skill_suggestions: List[str] = None) -> Thought:
        relevant_memories = memory_store.retrieve(f"{goal} {str(observation)}", top_k=3)
        memory_context = "\n".join([m.content for m in relevant_memories])

        skill_context = ""
        if skill_suggestions:
            skill_context = "\n## Skill Suggestions\n" + "\n".join([f"- {s}" for s in skill_suggestions])

        user_prompt = f"""Goal: {goal}
Step: {step}
Working memory: {self.working_memory}

Recent memories:
{memory_context}

Current observation:
{json.dumps(observation, indent=2)}

Analyze the current state and decide next actions.
"""

        try:
            system_prompt = REACT_SYSTEM_PROMPT.format(
                capabilities_json=json.dumps(capabilities, indent=2),
                goal=goal,
                step=step,
                max_steps=cfg.MAX_STEPS,
                action_history=[t.action_plan for t in self.thought_history[-10:]],
                memories=memory_context,
                skill_context=skill_context
            )

            image_path = observation.get("screenshot_labeled")
            thought_data = await self.llm.complete_chat(
                system_prompt,
                user_prompt,
                image_path=image_path
            )

            thought = Thought(
                step=step,
                reasoning=thought_data.get("reasoning", "No reasoning provided"),
                action_plan=thought_data.get("action_plan", []),
                confidence=float(thought_data.get("confidence", 0.0)),
                observation=observation
            )

        except Exception as e:
            logger.warning(f"LLM reason failed: {e}. Fallback used.")
            thought = Thought(
                step=step,
                reasoning=f"Error in reasoning: {e}. Falling back to basic observation.",
                action_plan=[{"capability": "get_ui", "params": {}}],
                confidence=0.5,
                observation=observation
            )

        self.thought_history.append(thought)
        return thought

    async def generate_summary(self, goal: str, status: str) -> str:
        """Generates a natural language summary of the mission's outcome and actions."""
        history_summary = []
        for t in self.thought_history:
            action_names = [a.get("capability") for a in t.action_plan]
            history_summary.append(f"Paso {t.step}: {t.reasoning} -> Acciones: {action_names} ({t.reflection})")

        history_text = "\n".join(history_summary)

        system_prompt = "Eres AndroMancer. Resume lo que has hecho para el usuario en su mismo idioma."
        user_prompt = f"""El usuario quería: {goal}
Resultado final: {status}

Historial de acciones:
{history_text}

Genera un mensaje natural y detallado (pero conciso) informando al usuario sobre el progreso de CADA sub-tarea mencionada en el objetivo.
Menciona explícitamente qué se logró y qué falló (si algo falló, explica brevemente por qué y que continuaste con lo demás).
Si el usuario preguntó en español, responde en español. Si preguntó en inglés, responde en inglés. Adapta el idioma.
Ejemplo: 'He enviado el mensaje de WhatsApp a tu mamá. En YouTube busqué el tutorial, pero no pude activar el modo PiP, así que continué con la nota en Keep y el login en LeetCode. ¡Todo lo demás está listo!'
"""

        try:
            return await self.llm.complete_text(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

        return f"Misión finalizada con estado: {status}."

    async def reflect(self, thought: Thought, result: Any) -> Thought:
        """Use the SMALL MODEL to validate UI after an action (Inference Optimization)"""
        success_str = "Success" if getattr(result, 'success', False) else "Failed"
        error_str = getattr(result, 'error', '')

        if not getattr(result, 'success', False):
            thought.reflection = f"Action Failed: {error_str}"
            return thought

        # If success, use Small Model to verify if the UI state changed as expected
        system_prompt = "Eres un experto en QA de Android. Valida si la acción realizada tuvo éxito basándote en el resultado técnico."
        user_prompt = f"Acción intentada: {json.dumps(thought.action_plan)}\nResultado: {success_str}\nError: {error_str}\n\n¿Parece que la acción logró su objetivo? Responde brevemente."

        try:
            validation = await self.llm.complete_text(
                system_prompt,
                user_prompt,
                model=cfg.SMALL_MODEL_NAME
            )
            thought.reflection = f"Action {success_str}. Validation: {validation}"
        except Exception:
            thought.reflection = f"Action {success_str}. {error_str}"

        return thought
