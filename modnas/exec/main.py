"""Run ModularNAS routines as main node."""
from modnas.utils.wrapper import run


def exec_main():
    """Run ModularNAS as main node."""
    override = [{
        'defaults': {
            'estim.*.main': True
        }
    }]
    return run(parse=True, override=override)


if __name__ == '__main__':
    exec_main()
