Contributing with AI:  The "Vibe Coding" Guide
==============================================


We welcome the use of AI assistants (like Claude, Gemini, Copilot, …) to
accelerate development, draft boilerplate, and explore architectures.
However, as an open-science project, we align our development practices
with the core values of the `Leiden Declaration on Artificial
Intelligence <https://leidendeclaration.ai/>`__.

When you “vibe code,” you must adhere to the following principles to
protect the integrity, openness, and reliability of our codebase.

1. Human Authorship and Ultimate Responsibility
-----------------------------------------------

AI may write the lines, but **you** are the author.

- **You own the correctness:** Do not trust AI-generated logic blindly,
  especially for complex scientific implementations like interatomic
  potentials or molecular dynamics workflows. The responsibility for the
  correctness and adequacy of the results remains exclusively with the
  human author.
- **Test everything:** AI-generated code must be accompanied by rigorous
  human-verified unit tests. “The AI said it works” is not a valid
  commit message.
- **No AI authors:** Credit and responsibility belong to humans;
  artificial intelligence may obscure, but does not replace, the
  collective human labor behind a result. Do not add AI tools as
  co-authors in your commits.

2. Transparent Disclosure
-------------------------

If an AI assistant generated a significant architectural component, a
complex algorithm, or an entire module, disclose it.

- **In Pull Requests:***(PR) Include a brief note in your PR description.
  *Example: “The boilerplate for the ``MLIPCalculator`` class and Pytest
  fixtures were generated using Claude 3.5, with manual refinements for
  CPU/GPU binding.”*
- **In Documentation:** If an AI tool was heavily relied upon for
  drafting documentation, mention it in a “Tool and computational
  resource disclosure” section.

3. Strict Attribution and Provenance
------------------------------------

AI models are known for synthesizing solutions without properly citing
the original human work they learned from.

- **Track the upstream:** The limitations of automated tools create a
  corresponding obligation for you to proactively find and credit the
  sources that made a new result possible. If the AI suggests a highly
  specific algorithm, track down the original source (e.g., a specific
  paper or an upstream library) and add proper citations in the code
  comments.
- **Respect licenses:** Ensure that the AI has not regurgitated
  proprietary or incompatibly licensed code (like GPL code into an
  MIT/Apache-licensed project).

4. Code Review
--------------

AI makes it easy to generate massive amounts of code quickly, which can
introduce material that makes peer reviewing significantly more
demanding.

- **Keep PRs focused:** Do not dump 2,000 lines of AI-generated code
  into a single PR. Break your submissions into logical, reviewable
  chunks.
- **Review your own code first:** Before requesting a review, read
  through the AI’s output. Remove redundant logic, fix weird naming
  conventions, and ensure the code matches our project’s style guide.
  Ensure the “vibe” translates into readable, maintainable Python.
- **No AI generated responses:** Code review is done by humans.
  When communicating with maintainers, do not generate responses
  using AI.

5. Open Science and Autonomy
----------------------------

We build tools for the whole community, not just those with access to
premium AI models. As research becomes more reliant on software, we must
adhere to the principles of open science.

- **No proprietary runtime dependencies:** The final, merged code must
  not depend on paid AI APIs or closed-source models to execute its core
  scientific functions.
- **Clarity over cleverness:** AI sometimes writes overly dense or
  “clever” code. Prioritize clarity and scientific understanding. The
  goal is maintainable open science, not just code that happens to
  compile.

Pull Request Checklist
----------------------

Before submitting your PR, please confirm:

- [ ] I have reviewed and understand all AI-generated code in this PR.
- [ ] I have added human-written tests for the new logic.
- [ ] I have properly cited any upstream algorithms or libraries the AI
  utilized.
- [ ] I have disclosed significant AI tool usage in the PR description.
