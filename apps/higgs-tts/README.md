# CogniVault

## Installation

```bash
pip install -r requirements.txt
```

## Run

### Using Docker (Recommended)

```bash
./run.sh
```

Or with a custom port:
```bash
PORT=50051 ./run.sh
```

### Without Docker (You need pytorch installed)

```bash
python watch_server.py --port 50051
```

## Credits

Based on [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)

