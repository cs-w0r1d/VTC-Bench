from qwen_agent.llm.schema import Message, ContentItem

# Test dict to Message conversion
msg_dict = {
    'role': 'user',
    'content': [
        {'image': '/path/to/image.jpg'},
        {'text': 'Hello'}
    ]
}

try:
    msg = Message(**msg_dict)
    print('Message created:', msg)
    print('Content type:', type(msg.content))
    print('Content:', msg.content)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
