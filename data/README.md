# Benchmark Datasets Directory

This directory contains benchmark datasets for evaluation. Each dataset should be in its own subdirectory with JSONL format files.

## Supported Datasets

- **MATH** - Competition mathematics problems
- **GSM8K** - Grade school math word problems  
- **AIME** - American Invitational Mathematics Examination
- **AIME25** - Recent AIME problems (2025)
- **AMC** - American Mathematics Competitions
- **OlympiadBench** - Mathematical olympiad problems
- **Minerva** - STEM reasoning problems
- **GPQA** - Graduate-level physics questions
- **MMLU-Pro** - Enhanced MMLU with challenging problems

## Dataset Format

Each JSONL file should contain one JSON object per line:

```json
{"problem": "What is 2+2?", "answer": "4"}
```

**Required Fields:**
- `problem` - The question or problem statement (string)
- `answer` - The correct answer (string)

**Optional Fields:** `id`, `difficulty`, `category`, `source`, `year`
