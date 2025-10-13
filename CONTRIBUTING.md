# Contributing to LLM Benchmark Evaluation

Thank you for considering contributing to this project! We welcome contributions of all kinds.

## Ways to Contribute

### 1. Report Bugs
- Use GitHub Issues to report bugs
- Include detailed reproduction steps
- Provide system information (OS, Python version, etc.)
- Include relevant error messages and logs

### 2. Suggest Features
- Open a GitHub Issue with the `enhancement` label
- Describe the feature and its use case
- Explain why it would be beneficial

### 3. Add New Datasets
To add support for a new dataset:
1. Create a new loader in `src/datasets/`
2. Inherit from `BaseDatasetLoader`
3. Implement required methods (`load_split`, `_standardize_format`)
4. Add tests
5. Update documentation

### 4. Improve Documentation
- Fix typos or unclear explanations
- Add examples
- Improve installation instructions
- Translate documentation

### 5. Submit Code

#### Development Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/llm_benchmark_eval.git
cd llm_benchmark_eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

#### Code Standards
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions focused and modular
- Add unit tests for new features

#### Pull Request Process
1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test your changes thoroughly
5. Commit with clear messages: `git commit -m "Add feature X"`
6. Push to your fork: `git push origin feature/my-feature`
7. Open a Pull Request

#### PR Requirements
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

### 6. Review Pull Requests
Help review open pull requests by:
- Testing the changes
- Providing constructive feedback
- Suggesting improvements

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Accept gracefully
- Prioritize community well-being

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other unprofessional conduct

## Questions?

Feel free to open a GitHub Issue for any questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

