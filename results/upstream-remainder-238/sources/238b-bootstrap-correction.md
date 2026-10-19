# 238b: correct the228 input-status check before any model work

Original238 stopped before creating its output or loading a model. It expected
the228 audit status to read`passed`; that scientific outcome record correctly
uses`complete` with separately retained numerical checks and forecast-order flag.
The original source/protocol and traceback are preserved.

238b changes only this bootstrap validation: require`complete`, the passed
global forecast-order flag and all retained checks within their recorded bounds.
It uses the identical inputs, operation list, rank rule, thresholds, computation,
numerical domains and interventions in the original238 protocol. No model or
intervention outcome was observed before this correction. It cannot turn228's
failed scientific hypothesis into a successful one.
