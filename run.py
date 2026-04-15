import sys
from src.engine import ResearchEngine

def main():
    print("\n" + "="*60)
    print("      🏥 HEALTHCARE FAIRNESS: MANUSCRIPT LAB 🏥")
    print("="*60)
    print("🚀 Initializing Automated Publication Pipeline...")
    
    engine = ResearchEngine()
    
    try:
        engine.run_all()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING RESEARCH RUN: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🏁 ALL ASSETS GENERATED. YOU CAN NOW CLOSE THIS WINDOW.")
    print("="*60)

if __name__ == "__main__":
    main()
