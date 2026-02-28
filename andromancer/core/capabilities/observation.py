import os
import re
import asyncio
import tempfile
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import List, Dict
from andromancer.core.capabilities.base import ADBCapability, Capability, ExecutionResult
from andromancer.utils.image import label_image
from andromancer.core.llm_client import AsyncLLMClient
from andromancer import config as cfg

logger = logging.getLogger("AndroMancer.Observation")

class UIScrapeCapability(ADBCapability, Capability):
    name = "get_ui"
    description = "Obtiene jerarquía UI actual, resumen de texto y captura etiquetada (SoM)"
    risk_level = "low"

    def __init__(self):
        self.llm = AsyncLLMClient()

    async def execute(self, use_cache: bool = False) -> ExecutionResult:
        try:
            temp_dir = Path(tempfile.gettempdir())

            if not os.access("/tmp", os.W_OK):
                temp_dir = Path.home() / ".cache" / "andromancer"

            temp_dir.mkdir(parents=True, exist_ok=True)
            local_ui_path = temp_dir / "ui.xml"
            local_screenshot_path = temp_dir / "screen.png"
            local_labeled_path = temp_dir / "screen_labeled.png"

            # 1. XML Dump
            await self._adb(["shell", "uiautomator", "dump", "/sdcard/ui.xml"])
            await self._adb(["pull", "/sdcard/ui.xml", str(local_ui_path)])

            # 2. Screenshot
            await self._adb(["shell", "screencap", "-p", "/sdcard/screen.png"])
            await self._adb(["pull", "/sdcard/screen.png", str(local_screenshot_path)])

            try:
                with open(local_ui_path, "r", encoding="utf-8") as f:
                    xml_content = f.read()

                root = ET.fromstring(xml_content)
                elements = self._parse_nodes(root)

                # Identify current package
                current_package = "unknown"
                if elements:
                    current_package = elements[0].get('package', 'unknown')

                # 3. SoM Labeling
                label_map = label_image(str(local_screenshot_path), elements[:25], str(local_labeled_path))

                # 4. Small Model Summarization (Inference Optimization)
                raw_summary = self._summarize_screen_basic(elements, current_package)
                screen_summary = await self._get_ai_summary(raw_summary, xml_content[:5000]) # Limit XML size

                return ExecutionResult(True, data={
                    "xml": xml_content,
                    "elements": elements,
                    "summary": screen_summary,
                    "current_package": current_package,
                    "screenshot_labeled": str(local_labeled_path),
                    "label_map": {str(k): v for k, v in label_map.items()}
                })
            except Exception as e:
                return ExecutionResult(False, error=f"XML parse error: {str(e)}")
        except Exception as e:
            return ExecutionResult(False, error=f"UI scrape error: {str(e)}")

    def _parse_nodes(self, root) -> List[Dict]:
        elements = []
        for node in root.iter('node'):
            if node.get('clickable') == 'true':
                elements.append({
                    "text": node.get('text', ''),
                    "content_desc": node.get('content-desc', ''),
                    "resource_id": node.get('resource-id', ''),
                    "class": node.get('class', ''),
                    "bounds": node.get('bounds', ''),
                    "package": node.get('package', '')
                })
        return elements

    async def _get_ai_summary(self, basic_summary: str, xml_snippet: str) -> str:
        """Use the SMALL MODEL to generate a cleaned UI summary"""
        system_prompt = "Eres un experto en analizar XML de Android. Resume la pantalla actual de forma concisa para un agente autónomo. Enfócate en elementos interactivos."
        user_prompt = f"Resumen básico: {basic_summary}\n\nFragmento XML:\n{xml_snippet}\n\nGenera un resumen limpio."

        try:
            return await self.llm.complete_text(
                system_prompt,
                user_prompt,
                model=cfg.SMALL_MODEL_NAME
            )
        except Exception as e:
            logger.warning(f"AI summary failed: {e}")
            return basic_summary

    def _summarize_screen_basic(self, elements: List[Dict], package: str = "unknown") -> str:
        summary_items = []
        for i, e in enumerate(elements[:25]):
            text = e['text'] or e['content_desc']
            if text:
                summary_items.append(f"[{i+1}] '{text}'")

        base = f"App: {package} | "
        if summary_items:
            return base + "Screen with: " + ", ".join(summary_items)
        return base + "Screen with no visible clickable elements"
