import os
import pytest

from ..config import Config, global_config
from ..apis import set_global_config, create_encoder
from .utilities import read_tasks_files

from unified_planning.io import PDDLReader

# Retrieve all tasks and encodings for parameterization
pddldir = os.path.join(os.path.dirname(__file__), "pddl")
tasks = read_tasks_files(pddldir)

@pytest.mark.parametrize("domainname, domainfile, problemfile, config_name", [
    (domainname, domainfile, problemfile, config_name)
    for domainname, domainfile, problemfile in tasks
    for config_name in Config().get_valid_configs().keys()  
])
def test_encoder_base_encode(domainname, domainfile, problemfile, config_name):
    """
    Test if each encoder configuration can successfully:
    1. Parse the PDDL files
    2. Create an encoder
    3. Call the base_encode function
    
    This verifies the basic encoding functionality works for each configuration.
    """
    # Create configuration with the specified encoding
    conf = Config()
    conf.set_config(config_name)
    conf.set('verbose', 0)  # Set low verbosity to reduce output
    set_global_config(conf)
    
    try:
        task = PDDLReader().parse_problem(domainfile, problemfile)
        encoder = global_config.get("encoder")
        compilationlist = global_config.get("compilationlist")
        encoder_instance, compiled_task = create_encoder(encoder, task, compilationlist)

        # Try to encode with horizon 0
        formula = encoder_instance.encode(0)
        
        # Verify the formula is not None
        assert formula is not None, f"Encoder {config_name} returned None from base_encode"
        
        # Check that formula has expected structure (dictionary with required keys)
        assert isinstance(formula, dict), f"Encoder {config_name} didn't return a formula dictionary"
        assert 'initial' in formula, f"Encoder {config_name} didn't include 'initial' in formula"
        assert 'goal' in formula, f"Encoder {config_name} didn't include 'goal' in formula"
        
        # The test passes if we reach this point without exceptions
        print(f"Encoder {config_name} successfully encoded {domainname}")
        
    except Exception as e:
        pytest.fail(f"Encoder {config_name} failed on {domainname}: {str(e)}")
