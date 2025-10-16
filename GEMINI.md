# Gemini Usage Guidelines

This document outlines best practices for interacting with Gemini, particularly concerning Git commits.

## Committing Changes

When prompted to commit changes, Gemini will:

1.  **Review Staged Changes:** Examine `git status` and `git diff --staged` to understand the modifications.
2.  **Propose Commit Message:** Generate a concise and informative commit message.
3.  **Use `-F` for Commit:** Utilize `git commit -F <file>` to apply the commit message from a temporary file.
    *   This ensures proper formatting and handling of multi-line messages.
4.  **Adhere to Line Length:** Commit message lines will not exceed 88 characters.
    *   The first line (subject) will be a concise summary.
    *   Subsequent lines in the body will provide more detail, wrapped at 88 characters.

**Example Commit Message Structure:**

```
Feat: Implement new feature or fix bug (max 88 chars)

- Detailed explanation of the first change, wrapped at 88 characters.
- Another detailed point, ensuring it adheres to the line length limit.
- Further context or reasoning for the changes made in this commit.
```
