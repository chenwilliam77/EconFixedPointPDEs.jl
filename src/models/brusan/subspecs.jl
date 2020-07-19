"""
```
init_subspec!(m::BruSan)
```
initializes alternative subspecs of the BruSan model
"""
function init_subspec!(m::BruSan)
    if subspec(m) == "ss0"
        return
    else
        error("Subspec $(subspec(m)) has not been defined.")
    end
end
