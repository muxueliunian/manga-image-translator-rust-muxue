---
name: Bug report
about: Create a report to help us improve
title: '[Bug]: '
labels: ["bug"]
assignees: ''

body:
  - type: textarea
    id: what-did
    attributes:
      label: Describe the bug
      description: A clear and concise description of what the bug is.
    validations:
      required: true

  - type: textarea
    id: cmdargs
    attributes:
      label: To Reproduce
      description: Steps to reproduce the behavior. Please use --verbose when running.
    validations:
      required: true
    
  - type: textarea
    id: exp-behavior
    attributes:
      label: Expected behavior
      description: A clear and concise description of what you expected to happen.
    validations:
      required: true
    
  - type: textarea
    id: behavior
    attributes:
      label: Actual behavior
      description: A clear and concise description of what actually happened.
    validations:
      required: true
  
  - type: textarea
    id: logs
    attributes:
      label: Console logs
      render: Shell

  - type: input
    id: version
    attributes:
      label: Software version
      description: Please provide the output of `--version`.
    validations:
      required: true
    
  - type: input
    id: os
    attributes:
      label: OS / Architecture
      description: Please provide your OS, version, and architecture (e.g., Linux x86_64, macOS arm64, Windows 11 x64).
    validations:
      required: true

  - type: textarea
    id: additional-context
    attributes:
      label: Additional context
      description: Add any other context about the problem here.
