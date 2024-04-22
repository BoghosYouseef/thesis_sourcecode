import typer
from rich import print as rprint
from PyInquirer import prompt, print_json, Separator
from utils.cli_utils import get_list_of_available_models, load_and_test_a_patch_model

app = typer.Typer()


@app.command("hi")
def sample1_func():
    rprint("[red bold]Hi[/red bold] [yellow]World[yello]")

@app.command("hello")
def sample_func():
    rprint("[red bold]Hello[/red bold] [yellow]World[yello]")

@app.command("load-model")
def load_model():
    questions = [
        {
            "type": "list",
            "name": "model",
            "message": "Select a model:",
            "choices": get_list_of_available_models(),
            "default": None,
        },
        {
            'type': 'input',
            'name': '3d-point',
            'message': 'input the coordinates of a 3d-point: ',
        }
    ]

    answer = prompt(questions=questions)
    predicted_patch = load_and_test_a_patch_model(answer['model'], answer['3d-point'])
    rprint(f"You have chosen model {answer['model']}")
    rprint(f"input point: {(answer['3d-point'])}")
    rprint(f"output: predicted patch = {predicted_patch}")


if __name__ == "__main__":
    app() 