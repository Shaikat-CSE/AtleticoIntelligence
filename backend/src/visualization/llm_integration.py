from typing import Dict, Any, Optional
import os
import httpx
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMIntegration:
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        api_base: str = None
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        self.model = model or os.getenv("GEMINI_MODEL")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_base = api_base or os.getenv("LLM_API_BASE", "").rstrip("/")

        if not self.api_key:
            logger.warning("[LLM] GEMINI_API_KEY not found in environment")

        logger.info(f"[LLM] Initialized - provider={self.provider}, model={self.model}, has_key={bool(self.api_key)}")

    def generate_explanation(self, structured_data: Dict[str, Any]) -> str:
        logger.info(f"[LLM] generate_explanation called with data: {structured_data}")

        if not self.api_key:
            logger.warning("[LLM] No API key, using fallback explanation")
            return self._generate_fallback_explanation(structured_data)

        if self.provider == "openai" and self.api_base:
            return self._call_openai_compatible(structured_data)
        elif self.provider == "gemini":
            return self._call_gemini(structured_data)

        logger.warning("[LLM] Invalid config, using fallback")
        return self._generate_fallback_explanation(structured_data)

    def _call_openai_compatible(self, structured_data: Dict[str, Any]) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        prompt = self._build_prompt(structured_data)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 256
        }

        logger.info(f"[LLM] Calling OpenAI-compatible API: {url}")

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                logger.info(f"[LLM] Response status: {response.status_code}")

                if response.status_code != 200:
                    logger.error(f"[LLM] API error: {response.text}")
                    return self._generate_fallback_explanation(structured_data)

                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    result = data["choices"][0]["message"]["content"]
                    logger.info(f"[LLM] Generated explanation: {result}")
                    return result
                else:
                    logger.warning("[LLM] No choices in response")
                    return self._generate_fallback_explanation(structured_data)

        except Exception as e:
            logger.error(f"[LLM] Exception: {type(e).__name__} - {str(e)}")
            return self._generate_fallback_explanation(structured_data)

    def _call_gemini(self, structured_data: Dict[str, Any]) -> str:
        if not self.model:
            logger.error("[LLM] GEMINI_MODEL not set")
            return self._generate_fallback_explanation(structured_data)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        prompt = self._build_prompt(structured_data)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 256
            }
        }

        logger.info(f"[LLM] Calling Gemini API: {url}")

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, params=params, json=payload)
                logger.info(f"[LLM] Response status: {response.status_code}")

                if response.status_code != 200:
                    logger.error(f"[LLM] API error: {response.text}")
                    return self._generate_fallback_explanation(structured_data)

                data = response.json()

                if "candidates" in data and len(data["candidates"]) > 0:
                    result = data["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"[LLM] Generated explanation: {result}")
                    return result
                else:
                    logger.warning("[LLM] No candidates in response")
                    return self._generate_fallback_explanation(structured_data)

        except Exception as e:
            logger.error(f"[LLM] Exception: {type(e).__name__} - {str(e)}")
            return self._generate_fallback_explanation(structured_data)

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        return f"""You are a soccer referee assistant explaining an offside decision.

Decision: {data.get('decision', 'UNKNOWN')}
Attacker Position: x={data.get('attacker_position', {}).get('x', 'N/A'):.1f}, y={data.get('attacker_position', {}).get('y', 'N/A'):.1f}
Defender Position: x={data.get('defender_position', {}).get('x', 'N/A'):.1f}, y={data.get('defender_position', {}).get('y', 'N/A'):.1f}
Confidence: {data.get('confidence', 0):.2f}

Provide a 2-3 sentence explanation of the offside decision in simple terms suitable for a TV broadcast.
"""

    def _generate_fallback_explanation(self, data: Dict[str, Any]) -> str:
        logger.info("[LLM] Using fallback explanation")
        decision = data.get("decision", "UNKNOWN")
        attacker_x = data.get("attacker_position", {}).get("x", 0)
        defender_x = data.get("defender_position", {}).get("x", 0)

        if decision == "OFFSIDE":
            result = (
                f"OFFSIDE - The attacker (x={attacker_x:.1f}) is beyond the second-last defender (x={defender_x:.1f}). "
                f"The attacker is in an offside position as they are closer to the goal line than the ball and the second-last defender."
            )
        else:
            result = (
                f"ONSIDE - The attacker (x={attacker_x:.1f}) is not beyond the second-last defender (x={defender_x:.1f}). "
                f"The attacker is level or behind the second-last defender when the ball was played."
            )

        logger.info(f"[LLM] Fallback explanation: {result}")
        return result


def generate_llm_explanation(
    data: Dict[str, Any],
    provider: str = None,
    model: str = None
) -> str:
    llm = LLMIntegration(provider, model)
    return llm.generate_explanation(data)
