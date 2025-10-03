

def agent_with_tool_stream_parser(stream,agent_responses=[],tool_responses=[],debug=False):
    for chunk in stream:
        message_chunk, metadata = chunk
        node_name = metadata.get('langgraph_node', 'unknown')
        if debug:
            print(message_chunk)
        if hasattr(message_chunk, 'content') and message_chunk.content:
            if node_name == 'agent':
                agent_responses.append(message_chunk.content)
            elif node_name == 'tools':
                tool_responses.append(message_chunk.content)
    return agent_responses,tool_responses