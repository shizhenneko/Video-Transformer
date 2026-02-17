import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.analyzer.models import KnowledgeDocument, AnalysisResult

def test_markdown_structure():
    # Create dummy data
    doc = KnowledgeDocument(
        title="Test Title",
        one_sentence_summary="Test Summary",
        key_takeaways=["Takeaway 1", "Takeaway 2"],
        deep_dive=[
            {"topic": "Topic 1", "explanation": "Exp 1", "example": "Ex 1"}
        ],
        glossary={"Term": "Def"},
        mind_map_structure=["root: Test", "  - Child"]
    )
    
    result = AnalysisResult(
        video_path="test.mp4",
        knowledge_doc=doc
    )
    
    md_output = result.to_markdown()
    
    print("Generated Markdown Preview (First 500 chars):")
    print("-" * 40)
    print(md_output[:500])
    print("-" * 40)
    
    # Check for Nano Banana Pro instruction
    nano_instruction = "Nano Banana Pro 作图指令"
    if nano_instruction in md_output:
        print(f"[PASS] Found '{nano_instruction}'")
    else:
        print(f"[FAIL] Missing '{nano_instruction}'")

    # Check order: Key Takeaways -> Mind Map -> Deep Dive
    try:
        idx_takeaways = md_output.index("## 📝 关键结论")
        idx_mindmap = md_output.index("## 🧠 知识蓝图") # Note: Title might change based on plan
        idx_deepdive = md_output.index("## 🔍 深度解析")
        
        if idx_takeaways < idx_mindmap < idx_deepdive:
            print("[PASS] Order is correct: Takeaways -> Mind Map -> Deep Dive")
        else:
            print("[FAIL] Order is incorrect!")
            print(f"Takeaways index: {idx_takeaways}")
            print(f"Mind Map index: {idx_mindmap}")
            print(f"Deep Dive index: {idx_deepdive}")
            
    except ValueError as e:
        print(f"[FAIL] Section missing: {e}")

if __name__ == "__main__":
    test_markdown_structure()
