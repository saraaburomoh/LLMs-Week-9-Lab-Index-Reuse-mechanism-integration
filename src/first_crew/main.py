#!/usr/bin/env python
import sys
import warnings
import json
import re
import os
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start, and_
from first_crew.crew import FirstCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# === Step 1: Define Flow State ===
class YelpRecommendationState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    user_profile: str = ""
    item_profile: str = ""
    web_research: str = ""
    raw_result: str = ""
    final_report: dict = {}

# === Step 2: Define the Parallel Flow ===
class YelpRecommendationFlow(Flow[YelpRecommendationState]):
    
    @start()
    def initialize_request(self):
        print(f"🚀 [Flow Start]: Initializing Parallel Flow for User: {self.state.user_id} | Item: {self.state.item_id}")
        return "initialized"

    @listen(initialize_request)
    def fetch_user_profile(self):
        print(f"👤 [Flow Action]: Analyzing User Profile in parallel...")
        crew_instance = FirstCrew().crew()
        
        # Filter to only the user task
        user_task = next(t for t in crew_instance.tasks if t.name == "analyze_user_task")
        crew_instance.tasks = [user_task]
        
        # We pass BOTH IDs to satisfy Agent variable interpolation, even if this task only uses one.
        result = crew_instance.kickoff(inputs={
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': '',
            'item_context': '',
            'web_context': ''
        })
        self.state.user_profile = result.raw
        return "user_ready"

    @listen(initialize_request)
    def fetch_item_profile(self):
        import time
        print(f"🏠 [Flow Action]: Waiting 10 seconds to stagger API requests...")
        time.sleep(10)
        print(f"🏠 [Flow Action]: Analyzing Item Profile in parallel...")
        crew_instance = FirstCrew().crew()
        
        # Filter to only the item task
        item_task = next(t for t in crew_instance.tasks if t.name == "analyze_item_task")
        crew_instance.tasks = [item_task]
        
        result = crew_instance.kickoff(inputs={
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': '',
            'item_context': '',
            'web_context': ''
        })
        self.state.item_profile = result.raw
        return "item_ready"

    @listen(fetch_item_profile)
    def fetch_web_research(self):
        import time
        print(f"🔍 [Flow Action]: Breathing for 10 seconds before Web Research...")
        time.sleep(10)
        print(f"🌐 [Flow Action]: Item profile ready. Now searching the web for real-time trends...")
        crew_instance = FirstCrew().crew()
        
        # Filter to only the web research task
        web_task = next(t for t in crew_instance.tasks if t.name == "web_research_task")
        crew_instance.tasks = [web_task]
        
        # We now have the item_profile, so the agent can see the business name!
        result = crew_instance.kickoff(inputs={
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': self.state.user_profile,
            'item_context': self.state.item_profile,
            'web_context': ''
        })
        self.state.web_research = result.raw
        return "web_ready"

    @listen(and_(fetch_user_profile, fetch_web_research))
    def run_final_prediction(self):
        import time
        print(f"⚖️ [Flow Action]: Data converged. Breathing for 5 seconds before Final Prediction...")
        time.sleep(5)
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").upper()
        print(f"⚖️ [Flow Action]: Kicking off {process_type} Crew for final prediction...")
        
        crew_instance = FirstCrew().crew()
        # Filter to only the prediction task
        predict_task = next(t for t in crew_instance.tasks if t.name == "predict_review_task")
        crew_instance.tasks = [predict_task]
        
        inputs = {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': self.state.user_profile,
            'item_context': self.state.item_profile,
            'web_context': self.state.web_research
        }
        
        result = crew_instance.kickoff(inputs=inputs)
        self.state.raw_result = result.raw
        return "prediction_completed"

    @listen(run_final_prediction)
    def process_and_save_results(self):
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").lower().replace("_report", "")
        filename = f"{process_type}_report.json"
        
        print(f"📊 [Flow Finalizing]: Parsing results and saving to {filename}...")
        report = self.extract_json_from_output(self.state.raw_result)
        self.state.final_report = report
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ [Flow Success]: Stars: {report.get('stars')} | Results saved.")
        return report

    def extract_json_from_output(self, raw_output: str) -> dict:
        """Extract and sanitize JSON from LLM raw output."""
        text = str(raw_output).strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        # Try to find anything between { and }
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Final fallback: Manual extraction of key fields
            stars_match = re.search(r'"stars":\s*([\d.]+)', text)
            review_match = re.search(r'"review":\s*"(.*)"', text, re.DOTALL)
            return {
                "stars": float(stars_match.group(1)) if stars_match else None,
                "review": review_match.group(1) if review_match else text,
                "_parse_manual": True
            }

def run():
    valid_modes = ["hierarchical", "collaborative", "sequential"]
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in valid_modes:
            os.environ["PROCESS_TYPE"] = arg
            print(f"🎯 [Mode Override]: Switching to {arg.upper()} mode.")

    test_json_path = "data/test_review_subset.json"
    if not os.path.exists(test_json_path):
        print(f"Error: {test_json_path} not found.")
        return

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Index 11 as requested
    first_case = test_data[11] if len(test_data) > 11 else test_data[0]

    flow = YelpRecommendationFlow()
    flow.state.user_id = first_case['user_id']
    flow.state.item_id = first_case['item_id']
    
    flow.kickoff()

def train(): pass
def replay(): pass
def test(): pass

if __name__ == "__main__":
    run()
