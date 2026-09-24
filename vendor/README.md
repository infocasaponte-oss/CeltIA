For vLLM releases where MiniCPM5 tool parsing is not included natively, place the official OpenBMB MiniCPM `tool_parsers/minicpm5xml_tool_parser.py` here and launch vLLM with `--enable-auto-tool-choice --tool-parser-plugin /workspace/vendor/minicpm5xml_tool_parser.py --tool-call-parser minicpm5`.

Check the current OpenBMB deployment documentation for whether your installed vLLM release still requires the bridge.
