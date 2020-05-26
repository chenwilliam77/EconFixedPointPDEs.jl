"""
```
init_subspec!(m::Li2020)
```
initializes alternative subspecs of the Li2020 model
"""
function init_subspec!(m::Li2020)
    if subspec(m) == "ss0"
        return
    else
        error("Subspec $(subspec(m)) has not been defined.")
    end
end
