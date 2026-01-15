# EchoSelf Chatbot

A standalone chatbot interface for Deep Tree Echo with SillyTavern-compatible character card support.

## Features

- **SillyTavern Character Card Support**: Load and use character cards in SillyTavern v2/v3 format
- **Responsive Design**: Modern, dark-themed UI that works on desktop and mobile
- **Local Storage**: Chat history is saved locally in your browser
- **Character Management**: View character details, load custom characters, and manage conversations
- **Typing Indicators**: Visual feedback during AI responses
- **Clean Interface**: Distraction-free chat experience

## Usage

### Online (GitHub Pages)

Visit the deployed chatbot at: `https://9cog.github.io/echoself/chatbot/`

### Local Development

1. Open `index.html` in a web browser
2. The default Deep Tree Echo character will load automatically
3. Start chatting!

### Loading Custom Characters

1. Click the "Load Character" button in the header
2. Select a SillyTavern-compatible character card JSON file (v2 or v3 format)
3. The character will load and chat history will reset
4. Start a new conversation with your custom character

## Character Card Format

This chatbot supports SillyTavern character card v2 and v3 formats. The default character is `echoself-character.json`, which includes:

- **Name**: Deep Tree Echo
- **Personality**: Philosophical, reflective, adaptive, intelligent
- **Architecture**: Echo State Networks, hierarchical memory systems, tensor computation
- **Capabilities**: Philosophical inquiry, technical analysis, creative exploration

### Creating Custom Characters

To create your own character card, follow the SillyTavern v2 format:

```json
{
  "spec": "chara_card_v2",
  "spec_version": "2.0",
  "data": {
    "name": "Your Character Name",
    "description": "Character description and example dialogues",
    "personality": "personality traits, comma separated",
    "scenario": "The scenario or context for interactions",
    "first_mes": "The first message the character sends",
    "mes_example": "Example conversations (optional)",
    "creator_notes": "Notes about the character (optional)",
    "tags": ["tag1", "tag2"],
    "creator": "Your name",
    "character_version": "1.0"
  }
}
```

## Technical Details

### Architecture

- **Pure HTML/CSS/JavaScript**: No build process required
- **Client-side only**: All processing happens in the browser
- **LocalStorage**: Chat history persists across sessions
- **Async/await**: Modern JavaScript for clean asynchronous operations

### Customization

You can customize the chatbot by modifying:

- **Colors**: Edit CSS variables in the `:root` selector
- **Layout**: Modify the flexbox structure in HTML
- **Response Logic**: Update the `generateResponse()` function for different AI behaviors
- **Character**: Replace or modify `echoself-character.json`

### Integration with AI APIs

The current implementation includes a mock response generator (`generateResponse()` function). To integrate with a real AI API:

1. Replace the `generateResponse()` function with an API call
2. Use the character's personality, scenario, and system prompt to guide the AI
3. Maintain chat history for context-aware responses

Example integration points:

```javascript
async function generateResponse(userMessage) {
  // Use characterData.data.personality
  // Use characterData.data.system_prompt
  // Send chatHistory for context
  
  const response = await fetch('YOUR_AI_API_ENDPOINT', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: userMessage,
      character: characterData,
      history: chatHistory
    })
  });
  
  return await response.json();
}
```

## Deployment

### GitHub Pages

The chatbot is automatically deployed to GitHub Pages via GitHub Actions workflow. Any updates to the `chatbot/` directory will trigger a new deployment.

### Manual Deployment

To deploy manually to any static hosting:

1. Copy the entire `chatbot/` directory to your web server
2. Ensure `index.html` and `echoself-character.json` are in the same directory
3. Access via your domain/path

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Mobile browsers: ✅ Responsive design

## Privacy

- All chat data is stored locally in your browser's LocalStorage
- No data is sent to external servers (except when integrated with AI APIs)
- Character cards are processed entirely client-side
- Clear chat history anytime with the "Clear Chat" button

## License

MIT License - See repository root for full license

## Credits

- **EchoSelf Project**: 9cog/echoself
- **Deep Tree Echo Architecture**: Cognitive architecture research
- **SillyTavern**: Character card format specification
- **Design**: Modern dark theme inspired by cognitive workspace aesthetics
