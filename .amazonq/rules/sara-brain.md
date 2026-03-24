# Sara Brain Connection

The persistent Sara Brain database lives at:

```
C:\Users\pearljen\.sara_brain\sara.db
```

When the user asks to "use sara", "connect to sara", "interface with sara", or similar:

1. Connect to the DB at the path above using `Brain(r'C:\Users\pearljen\.sara_brain\sara.db')`
2. Import: `from sara_brain.core.brain import Brain`
3. Always run from the repo root: `c:\Users\pearljen\Documents\me\repo\SaraBrain`
4. This is a persistent brain — all teaches, associations, and categories survive across sessions
5. Act as Sara's interface: use brain.teach(), brain.recognize(), brain.why(), brain.trace(), brain.stats(), etc.
