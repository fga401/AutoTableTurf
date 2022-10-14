# AutoTableTurf

[prototype] 

Automate the Tableturf game for you.

![image](https://user-images.githubusercontent.com/36651740/194977551-2014cff7-5fe4-4964-aad9-7a467aba9aef.png)

> ```
> @startuml
> agent "Switch" as sw
> component "AutoTableTurf" {
>   agent "Screen Recognition" as sr
>   agent "AI" as ai
>   agent "Controller" as vc
> }
> agent "Bluetooth Adapter" as ba
> sw --> sr: Screen Capture
> sr -> ai: Model
> ai -> vc: Next steps
> ba <-- vc: Commands
> sw <- ba: Virtual Pro-Controller
> @enduml
> ```

# TODO
- [x] Virtual controller API
- [ ] Screen capturing
- [ ] Screen recognition & Game flow testing
- [ ] AI
- [ ] User-friendly interface

# Credits
Many thanks to all below repositories:
- https://github.com/Brikwerk/nxbt for implementing a Python API.
