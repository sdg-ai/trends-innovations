import subprocess


def main():
    command = [
        "uvicorn",
        "app:app",
        "--use-colors",
        "--log-level=info"
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()