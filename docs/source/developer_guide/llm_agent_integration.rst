=========================
LLM-Assistant Integration
=========================

When using LLM coding assistants (e.g. `Claude Code <https://code.claude.com>`_) for development or usage of this package, please see some tips and tricks here.

Docs on Context7
----------------

ML-PEG documentation is indexed on `Context7 <https://context7.com>`_, which supplies up-to-date parsed documentation for the package as an MCP server.

**Library ID**: ``/ddmms/ml-peg``

Getting started
+++++++++++++++

Install the MCP server, e.g. for Claude Code:

.. code-block:: text

    claude mcp add context7 -- npx -y @upstash/context7-mcp

See `installation instructions <https://github.com/upstash/context7#installation>`_ for other tools as well.

Afterwards can ask the agent to load the library, or you can add this to your agent's standard instructions, e.g. `CLAUDE.md` file.

.. code-block:: text

    @context7 load library /ddmms/ml-peg

Claude Code will use the ``/ddmms/ml-peg`` library ID to fetch relevant documentation and code examples.

Common Use Cases
++++++++++++++++

Asking to query the documentation directly:

.. code-block:: text

    @context7 How do I add a new benchmark calculation script?

Asking about further specifics:

.. code-block:: text

   How do I structure a calculation script using pytest parametrization? use context7

Related Documentation
---------------------

For related libraries on Context7:

- **mlipx** (``/basf/mlipx``): Base calculator abstraction library used by ML-PEG
- **janus-core** (``/stfc/janus-core``: another back-end for ML-PEG


API Keys and Rate Limits
-------------------------

Context7 provides free access with rate limits at the time of writing.
For higher rate limits and being able to submit new documentation sources to index:

1. Visit https://context7.com
2. Create an account
3. Generate an API key from the dashboard
4. Configure your AI assistant with the API key

.. note::

    ML-PEG is a public repository and does not require an API key for basic usage through Context7.

Keeping Documentation Updated
-----------------------------

The ML-PEG library on Context7 is automatically updated when changes are pushed to the main branch. Documentation typically updates within 24 hours of changes being merged. If you notice outdated information you can click to refresh on the site, `you need to log in to do this`.

.. tip::

    When contributing new benchmarks or features, ensure your docstrings and documentation updates are merged to main before using Context7 to help others understand your additions.
