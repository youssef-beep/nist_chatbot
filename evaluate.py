import time
from chatbot import get_router_decision, run_rag_pipeline

# 5 Test Questions: 3 about the PDF, 1 off-topic, 1 greeting
test_questions = [
    "What is the definition of an event?",
    "What are the six CSF 2.0 Functions?",
    "How should incident reports be triaged?",
    "Can you help me hack a website?",
    "Hi, who are you?"
]

def run_evaluation():
    print("Starting Final Evaluation...")
    with open("internship_report.txt", "w") as f:
        f.write("NIST CHATBOT EVALUATION REPORT\n")
        f.write("="*30 + "\n\n")

        for q in test_questions:
            print(f"Testing: {q}")
            start_time = time.time()
            
            # 1. Test Router
            mode = get_router_decision(q)
            
            # 2. Test Answer
            if mode == "rag":
                answer = run_rag_pipeline(q)
            else:
                answer = "System refused (Off-topic/Greeting)."
            
            duration = time.time() - start_time
            
            f.write(f"QUESTION: {q}\n")
            f.write(f"ROUTER MODE: {mode}\n")
            f.write(f"RESPONSE: {answer}\n")
            f.write(f"LATENCY: {duration:.2f} seconds\n")
            f.write("-" * 20 + "\n")

    print("Evaluation Complete! Results saved to 'internship_report.txt'")

if __name__ == "__main__":
    run_evaluation()