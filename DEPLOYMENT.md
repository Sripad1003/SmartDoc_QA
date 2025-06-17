# Deployment Guide

## Quick Deploy Options

### 1. 🚀 Railway (Recommended)
1. Fork this repository
2. Connect to Railway
3. Set environment variable: `GEMINI_API_KEY`
4. Deploy automatically

### 2. 🐳 Docker
\`\`\`bash
# Build and run
docker build -t qa-system .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key qa-system
\`\`\`

### 3. 🌐 Render
1. Connect your GitHub repository
2. Set environment variables in dashboard
3. Deploy

### 4. ⚡ Vercel
1. Connect your GitHub repository  
2. Set environment variables in dashboard
3. Deploy

## Local Development

### Prerequisites
- Python 3.11+
- Git

### Setup
1. Clone repository:
   \`\`\`bash
   git clone <your-repo-url>
   cd qa-system
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Set environment variables:
   \`\`\`bash
   cp .env.example .env
   # Edit .env with your Gemini API key
   \`\`\`

4. Run the system:
   \`\`\`bash
   python start.py
   \`\`\`

5. Access the application:
   - Frontend: http://127.0.0.1:8501
   - API: http://127.0.0.1:8000
   - API Docs: http://127.0.0.1:8000/docs

## Environment Variables

### Required
- `GEMINI_API_KEY`: Your Google Gemini API key

### Optional
- `ENVIRONMENT`: `development` or `production`
- `DEBUG`: `true` or `false`
- `MAX_CHUNK_SIZE`: Document chunk size (default: 1500)
- `MAX_ANSWER_LENGTH`: Maximum answer length (default: 2000)

## Production Considerations

### Security
- Set specific `ALLOWED_ORIGINS` instead of `*`
- Use HTTPS in production
- Set `DEBUG=false`
- Set `ENVIRONMENT=production`

### Performance
- Use multiple workers for high traffic
- Consider Redis for caching (future enhancement)
- Monitor response times and adjust timeouts

### Monitoring
- Check `/health` endpoint for system status
- Monitor `/system-stats` for performance metrics
- Review logs for errors and performance issues

## Troubleshooting

### Common Issues
1. **API key not working**: Check environment variable is set correctly
2. **File upload fails**: Check file size limits and format support
3. **Slow responses**: First-time embedding generation takes longer
4. **Import errors**: Install missing packages with pip

### Getting Help
1. Check system logs
2. Test with sample content first
3. Use document test feature to debug processing issues
4. Verify API key has proper permissions
\`\`\`

Now let me update the README to reflect the clean structure:
