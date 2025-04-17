from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
import concurrent.futures
import textwrap
from groq import Groq


class BaseRecommender(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def recommend_visualizations(self, n: int = 3) -> pd.DataFrame:
        pass


class LLMVisualRecommender(BaseRecommender):
    def __init__(self, df: pd.DataFrame, api_keys: dict):
        super().__init__(df)
        self.api_keys = api_keys
        self.clients = {
            'groq': Groq(api_key=self.api_keys.get("groq"))
        }
        self.model_ids = {
            'llama3-70b': 'llama3-70b-8192',
            'llama3-8b': 'llama3-8b-8192',
            'llama3-versatile': 'llama-3.3-70b-versatile'
        }

    def recommend_visualizations(self, n: int = 3) -> pd.DataFrame:
        prompt = self._create_prompt(self._describe_dataframe())
        responses = self._query_all_models(prompt)
        parsed = [rec for resp in responses for rec in self._parse_llm_response(resp)]
        unique = self._deduplicate(parsed)
        return pd.DataFrame(unique[:n])

    def _describe_dataframe(self) -> str:
        desc = [
            f"DataFrame Shape: {self.df.shape}",
            f"Columns: {', '.join(self.df.columns)}"
        ]
        return '\n'.join(desc)

    def _create_prompt(self, df_description: str) -> str:
        return textwrap.dedent(f"""
        You are a data visualization expert analyzing this dataset:

        {df_description}

        Recommend 3 insightful visualizations for exploring this data.
        For each recommendation, provide:
        Plot Type: <type>
        Variables: <var1, var2, ...>
        Separate recommendations with '---'
        """)

    def _query_all_models(self, prompt: str) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._query_llm, model_id, prompt)
                for model_id in self.model_ids.values()
            ]
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    def _query_llm(self, model_id: str, prompt: str) -> str:
        response = self.clients['groq'].chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def _parse_llm_response(self, response: str) -> List[Dict]:
        results = []
        for block in response.split('---'):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines or not lines[0].lower().startswith("plot type"):
                continue
            try:
                plot_type = lines[0].split(":", 1)[1].strip().lower()
                variables = [v.strip() for v in lines[1].split(":", 1)[1].split(',')]
                valid_vars = [v for v in variables if v in self.df.columns]
                if valid_vars:
                    results.append({
                        "Plot_Type": plot_type,
                        "Variables": ', '.join(valid_vars)
                    })
            except Exception:
                continue
        return results

    def _deduplicate(self, items: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for item in items:
            key = (item['Plot_Type'], item['Variables'])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
