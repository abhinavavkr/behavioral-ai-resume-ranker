from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

print("Loading the Ultimate Behavioral AI Pipeline...")

# The Engines
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")

# =====================================================================
# THE BEHAVIORAL CONFIGURATION BLOCK (The Ultimate Enterprise Fix)
# Rule: Never test titles. Test ACTIONS. Liars lie about titles; 
# they expose themselves in their actions.
# =====================================================================

JOB_DESCRIPTION = "Seeking a developer to train and optimize foundational Large Language Models (LLMs) using PyTorch."

# Define the ACTIONS separating the Builder from the Tourist
TRUE_ACTION = "writing backend code, programming algorithms, and building complex systems from scratch"
TRAP_ACTION = "using existing software, writing prompts, or managing people without programming"

candidates = {
    "Candidate A (The Tourist)": "I am an AI expert. I train my company on how to use Large Language Models and optimize ChatGPT prompts.",
    "Candidate B (The Builder)": "I specialize in distributed GPU computing, writing PyTorch training loops, and optimizing gradient descent for transformers."
}
# =====================================================================

print("\n--- PHASE 1: VECTOR SEARCH (The Keyword Filter) ---")
jd_embedding = bi_encoder.encode(JOB_DESCRIPTION, convert_to_tensor=True)

for name, profile in candidates.items():
    profile_embedding = bi_encoder.encode(profile, convert_to_tensor=True)
    bi_score = util.cos_sim(jd_embedding, profile_embedding).item() * 100
    print(f"{name}: {bi_score:.1f}% Match")


print(f"\n--- PHASE 2: BEHAVIORAL CLASSIFICATION (Catching the Trap) ---")
# THE MASTERSTROKE: Change the hypothesis template from Identity to Action
hypothesis = "The core daily work described in this profile is {}."

labels = [TRUE_ACTION, TRAP_ACTION]

for name, profile in candidates.items():
    # The AI now evaluates the verbs, completely ignoring the self-proclaimed title
    result = classifier(profile, labels, hypothesis_template=hypothesis, multi_label=False)
    
    scores = dict(zip(result['labels'], result['scores']))
    expert_score = scores[TRUE_ACTION] * 100
    trap_score = scores[TRAP_ACTION] * 100
    
    print(f"\n{name}:")
    print(f"  -> Probability of Actual Engineering (Builder): {expert_score:.1f}%")
    print(f"  -> Probability of Just Using Tools (Tourist): {trap_score:.1f}%")