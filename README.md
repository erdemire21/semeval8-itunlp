# semeval8-itunlp

## Configuration

This project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```env
# API Configuration
API_KEY=your_api_key_here
API_BASE_URL=your_api_base_url_here

# LLM Model Configuration
MAIN_LLM=deepseek-ai/DeepSeek-R1
ERROR_LLM=deepseek-ai/DeepSeek-R1
```

### Environment Variables

- `API_KEY`: Your API key for accessing the LLM services
- `API_BASE_URL`: The base URL for the API endpoint
- `MAIN_LLM`: The model to use for primary code generation (defaults to "deepseek-ai/DeepSeek-R1")
- `ERROR_LLM`: The model to use for error correction (defaults to "deepseek-ai/DeepSeek-R1")