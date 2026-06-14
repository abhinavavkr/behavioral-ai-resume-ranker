# behavioral-ai-resume-ranker
A lightweight, enterprise-grade NLP pipeline that replaces broken keyword-matching ATS filters with Behavioral Zero-Shot Classification.
# Universal Behavioral AI Resume Ranker 🤖💼

An open-source, enterprise-grade hybrid NLP pipeline that fixes broken keyword-matching ATS filters using **Behavioral Zero-Shot Classification**. 

## 🚀 The Core Problem
Traditional Applicant Tracking Systems (ATS) rely on simple keyword vector similarity. They are completely gamed by candidates who stuff their profiles with buzzwords without having the actual technical skills ("Tech Tourists"), while filtering out real experts who express capability using deep industry jargon ("The Builders").

## 🧠 The Solution (Behavioral vs Identity)
Instead of classifying *identity titles* (which can be lied about), this architecture abstracts candidate evaluation down to **Behavioral Actions**. Utilizing a localized Natural Language Inference (NLI) model (`deberta-v3-small`), the system analyzes the verbs and context to mathematically separate true hands-on technical work from superficial tool usage or administrative management.

## 🛠️ How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/abhinavavkr/behavioral-ai-resume-ranker.git](https://github.com/abhinavavkr/behavioral-ai-resume-ranker.git)
