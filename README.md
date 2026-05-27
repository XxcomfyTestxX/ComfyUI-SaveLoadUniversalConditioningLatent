
# Universal Load/SaveConditioning and Load/SaveLatent

BASED ON https://github.com/endman100/ComfyUI-SaveAndLoadPromptCondition

## SaveConditioning Node

### Description
The `SaveConditioning` node is designed to save conditioning data to binary files. This is useful for storing and reusing conditioning information across different sessions or applications.

### Input Types
- **conditionings**: A list of tuples where each tuple contains text embeds data and a dictionary with a "pooled_output" key. //SDXL

## LoadConditioning Node

### Description
The `LoadConditioning` node is designed to load conditioning data from binary files. This allows for the reuse of previously saved conditioning information.

### Return Types
- **conditioning**: A list of conditioning data.

## SaveLatent Node

### Description
The `SaveLatent` node is designed to save conditioning data to binary files. This is useful for storing and reusing conditioning information across different sessions or applications.

### Input Types
- **conditionings**: A latent to be loaded in another session.

## LoadConditioning Node

### Description
The `LoadLatent` node is designed to load a previously saved latent. Has further support than Comfy's native one, which errors on LTX.

### Return Types
- **latent**: A latent image/video to decode.
