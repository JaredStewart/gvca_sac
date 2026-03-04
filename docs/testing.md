# Testing Guide

## Overview

The project uses pytest for backend testing and vitest for frontend testing.

## Backend Testing

### Setup

```bash
cd backend
uv sync --dev  # Install dev dependencies
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/routers/test_tagging.py

# Run specific test
uv run pytest tests/routers/test_tagging.py::TestTagDistribution::test_distribution_percentage_calculation
```

### Test Structure

```
backend/tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── pytest.ini           # Pytest configuration
├── test_config.py       # Config tests
├── routers/
│   ├── test_tagging.py  # Tagging endpoint tests
│   └── test_data.py     # Data endpoint tests
├── services/
│   └── test_job_queue.py
└── core/
    └── test_tagging.py
```

### Key Fixtures

Located in `conftest.py`:

- `mock_pb_client`: Mock PocketBase client
- `mock_openai_client`: Mock OpenAI client
- `sample_responses`: Test free response data
- `sample_tagging_results`: Test tagging data
- `test_client`: FastAPI TestClient

### Writing Backend Tests

```python
import pytest
from unittest.mock import patch, AsyncMock

class TestMyFeature:
    @pytest.mark.asyncio
    async def test_something(self, mock_pb_client):
        with patch("app.routers.my_router.pb_client", mock_pb_client):
            mock_pb_client.get_list.return_value = {"items": []}
            # Test code here
```

## Frontend Testing

### Setup

```bash
cd frontend
npm install  # Installs vitest and testing-library
```

### Running Tests

```bash
# Run all tests
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage
```

### Test Structure

```
frontend/src/
├── __tests__/
│   └── setup.ts         # Test setup and mocks
└── components/
    └── tagging/
        └── __tests__/
            └── TagCell.test.tsx
```

### Writing Frontend Tests

```typescript
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MyComponent } from '../MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })

  it('handles click events', () => {
    const onClick = vi.fn()
    render(<MyComponent onClick={onClick} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalled()
  })
})
```

## Test Coverage Goals

- Critical paths: 80%+ coverage
- Bug fixes: Add test that reproduces the bug
- New features: Test happy path and edge cases
