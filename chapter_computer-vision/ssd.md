# Object Detection Using Convolutional Neural Networks

So far, when we've talked about making predictions based on images,
we were concerned only with classification.
We asked questions like is this digit a "0", "1", ..., or "9?"
or, does this picture depict a "cat" or a "dog"?
Object detection is a more challenging task.
Here our goal is not only to say *what* is in the image
but also to recognize *where* it is in the image.
As an example, consider the following image, which depicts two dogs and a cat together with their locations.

![](../img/dogdogcat.png)

So object defers from image classification in a few ways.
First, while a classifier outputs a single category per image,
an object detector must be able to recognize multiple objects in a single image.
Technically, this task is called *multiple object detection*,
but most research in the area addresses the multiple object setting,
so we'll abuse terminology just a little.
Second, while classifiers need only to output probabilities over classes,
object detectors must output both probabilities of class membership
and also the coordinates that identify the location of the objects.


On this chapter we'll demonstrate the single shot multiple box object detector (SSD),
a popular model for object detection that was first described in [this paper](https://arxiv.org/abs/1512.02325),
and is straightforward to implement in MXNet Gluon.


## SSD:  Single Shot MultiBox Detector

The SSD model predicts anchor boxes at multiple scales. The model architecture is illustrated in the following figure.

![](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj48c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGw9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMzYzIDI3NyA1MTEgMjY5IiB3aWR0aD0iNTExcHQiIGhlaWdodD0iMjY5cHQiIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyI+PG1ldGFkYXRhPiBQcm9kdWNlZCBieSBPbW5pR3JhZmZsZSA2LjYuMSA8ZGM6ZGF0ZT4yMDE3LTA5LTEyIDE4OjU5OjAxICswMDAwPC9kYzpkYXRlPjwvbWV0YWRhdGE+PGRlZnM+PG1hcmtlciBvcmllbnQ9ImF1dG8iIG92ZXJmbG93PSJ2aXNpYmxlIiBtYXJrZXJVbml0cz0ic3Ryb2tlV2lkdGgiIGlkPSJTaGFycEFycm93X01hcmtlciIgdmlld0JveD0iLTQgLTQgMTAgOCIgbWFya2VyV2lkdGg9IjEwIiBtYXJrZXJIZWlnaHQ9IjgiIGNvbG9yPSJibGFjayI+PGc+PHBhdGggZD0iTSA1IDAgTCAtMyAtMyBMIDAgMCBMIDAgMCBMIC0zIDMgWiIgZmlsbD0iY3VycmVudENvbG9yIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9nPjwvbWFya2VyPjxmb250LWZhY2UgZm9udC1mYW1pbHk9IkhlbHZldGljYSBOZXVlIiBmb250LXNpemU9IjE2IiBwYW5vc2UtMT0iMiAwIDUgMyAwIDAgMCAyIDAgNCIgdW5pdHMtcGVyLWVtPSIxMDAwIiB1bmRlcmxpbmUtcG9zaXRpb249Ii0xMDAiIHVuZGVybGluZS10aGlja25lc3M9IjUwIiBzbG9wZT0iMCIgeC1oZWlnaHQ9IjUxNyIgY2FwLWhlaWdodD0iNzE0IiBhc2NlbnQ9Ijk1MS45OTU4NSIgZGVzY2VudD0iLTIxMi45OTc0NCIgZm9udC13ZWlnaHQ9IjUwMCI+PGZvbnQtZmFjZS1zcmM+PGZvbnQtZmFjZS1uYW1lIG5hbWU9IkhlbHZldGljYU5ldWUiLz48L2ZvbnQtZmFjZS1zcmM+PC9mb250LWZhY2U+PGxpbmVhckdyYWRpZW50IHgxPSIwIiB4Mj0iMSIgaWQ9IkdyYWRpZW50IiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjZGFkYWRhIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjYTVhNWE1Ii8+PC9saW5lYXJHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9Ik9ial9HcmFkaWVudCIgeGw6aHJlZj0iI0dyYWRpZW50IiBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKDQzMi4yMjEzIDM5Mi41Njc3Nykgcm90YXRlKC0xNzgpIHNjYWxlKDYxLjE5MTMpIi8+PGxpbmVhckdyYWRpZW50IHgxPSIwIiB4Mj0iMSIgaWQ9IkdyYWRpZW50XzIiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj48c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiNkYWRhZGEiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNkYWRhZGEiLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iT2JqX0dyYWRpZW50XzIiIHhsOmhyZWY9IiNHcmFkaWVudF8yIiBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKDQ0OS4zNDA3NiA0MTcuNzgzNDUpIHJvdGF0ZSgtMTc4KSBzY2FsZSgyMy40MDcxODIpIi8+PGxpbmVhckdyYWRpZW50IGlkPSJPYmpfR3JhZGllbnRfMyIgeGw6aHJlZj0iI0dyYWRpZW50XzIiIGdyYWRpZW50VHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDQ3LjUwMjkyIDMxNS4xNjIxMykgcm90YXRlKC0xNzgpIHNjYWxlKDczLjc2MjE5KSIvPjxsaW5lYXJHcmFkaWVudCBpZD0iT2JqX0dyYWRpZW50XzQiIHhsOmhyZWY9IiNHcmFkaWVudCIgZ3JhZGllbnRUcmFuc2Zvcm09InRyYW5zbGF0ZSg3MzcuMjk4OTcgNDM1LjMyNzk0KSByb3RhdGUoLTE3OCkgc2NhbGUoMzYuODcwMzk1KSIvPjxsaW5lYXJHcmFkaWVudCBpZD0iT2JqX0dyYWRpZW50XzUiIHhsOmhyZWY9IiNHcmFkaWVudF8yIiBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKDc0OC4xNzAzIDQ0Ni4xNzE4Nikgcm90YXRlKC0xNzgpIHNjYWxlKDEzLjU5ODg2MykiLz48bGluZWFyR3JhZGllbnQgaWQ9Ik9ial9HcmFkaWVudF82IiB4bDpocmVmPSIjR3JhZGllbnRfMiIgZ3JhZGllbnRUcmFuc2Zvcm09InRyYW5zbGF0ZSg3NDcuMzY0OTggNDAxLjczMzAzKSByb3RhdGUoLTE3OCkgc2NhbGUoNDUuNzU3ODI2KSIvPjxsaW5lYXJHcmFkaWVudCBpZD0iT2JqX0dyYWRpZW50XzciIHhsOmhyZWY9IiNHcmFkaWVudCIgZ3JhZGllbnRUcmFuc2Zvcm09InRyYW5zbGF0ZSg1ODAuNzMyNDYgNDE4LjkzODU0KSByb3RhdGUoLTE3OCkgc2NhbGUoNTIuMjQ2NzQ0KSIvPjxsaW5lYXJHcmFkaWVudCBpZD0iT2JqX0dyYWRpZW50XzgiIHhsOmhyZWY9IiNHcmFkaWVudF8yIiBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKDU5NS44NzM4MyA0MzYuMzY3MjgpIHJvdGF0ZSgtMTc4KSBzY2FsZSgxOS41MDk1NTEpIi8+PGxpbmVhckdyYWRpZW50IGlkPSJPYmpfR3JhZGllbnRfOSIgeGw6aHJlZj0iI0dyYWRpZW50XzIiIGdyYWRpZW50VHJhbnNmb3JtPSJ0cmFuc2xhdGUoNTk0LjU4OTQ0IDM2NS4xNDc0Mykgcm90YXRlKC0xNzgpIHNjYWxlKDY0LjIxODAwNCkiLz48L2RlZnM+PGcgc3Ryb2tlPSJub25lIiBzdHJva2Utb3BhY2l0eT0iMSIgc3Ryb2tlLWRhc2hhcnJheT0ibm9uZSIgZmlsbD0ibm9uZSIgZmlsbC1vcGFjaXR5PSIxIj48dGl0bGU+Q2FudmFzIDE8L3RpdGxlPjxyZWN0IGZpbGw9IndoaXRlIiB3aWR0aD0iOTM3IiBoZWlnaHQ9IjYxOSIvPjxnPjx0aXRsZT5MYXllciAxPC90aXRsZT48bGluZSB4MT0iNzQ2LjMyODg2IiB5MT0iNDI1LjMyODg2IiB4Mj0iODIxLjEwMDA1IiB5Mj0iNDI1LjYxNzY1IiBtYXJrZXItZW5kPSJ1cmwoI1NoYXJwQXJyb3dfTWFya2VyKSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48bGluZSB4MT0iNzIwIiB5MT0iMzk4Ljk1OTczIiB4Mj0iNzk0Ljc3MTIiIHkyPSIzOTguNjcwOTQiIG1hcmtlci1lbmQ9InVybCgjU2hhcnBBcnJvd19NYXJrZXIpIiBzdHJva2U9ImJsYWNrIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0cm9rZS13aWR0aD0iMSIvPjx0ZXh0IHRyYW5zZm9ybT0idHJhbnNsYXRlKDczNi4xOTggMzc0Ljc3NikiIGZpbGw9ImJsYWNrIj48dHNwYW4gZm9udC1mYW1pbHk9IkhlbHZldGljYSBOZXVlIiBmb250LXNpemU9IjE2IiBmb250LXdlaWdodD0iNTAwIiB4PSIuMjU2IiB5PSIxNSIgdGV4dExlbmd0aD0iNTYiPmNsYXNzIHByPC90c3Bhbj48dHNwYW4gZm9udC1mYW1pbHk9IkhlbHZldGljYSBOZXVlIiBmb250LXNpemU9IjE2IiBmb250LXdlaWdodD0iNTAwIiB4PSI1NS45NjgiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI0OS43NzYiPmVkaWN0b3I8L3RzcGFuPjwvdGV4dD48dGV4dCB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3NjQgNDAyLjQ0NzE0KSIgZmlsbD0iYmxhY2siPjx0c3BhbiBmb250LWZhbWlseT0iSGVsdmV0aWNhIE5ldWUiIGZvbnQtc2l6ZT0iMTYiIGZvbnQtd2VpZ2h0PSI1MDAiIHg9Ii4xNDQiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI0Ni4yMjQiPmJveCBwcjwvdHNwYW4+PHRzcGFuIGZvbnQtZmFtaWx5PSJIZWx2ZXRpY2EgTmV1ZSIgZm9udC1zaXplPSIxNiIgZm9udC13ZWlnaHQ9IjUwMCIgeD0iNDYuMDgiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI0OS43NzYiPmVkaWN0b3I8L3RzcGFuPjwvdGV4dD48cGF0aCBkPSJNIDM3NC42NDQzIDI4OCBMIDM3NC42NDQzIDQ0My4yNSBMIDQyOC42NDQzIDQ5NSBMIDQyOC42NDQzIDMzOS43NSBaIiBmaWxsPSJ1cmwoI09ial9HcmFkaWVudCkiLz48cGF0aCBkPSJNIDM3NC42NDQzIDI4OCBMIDM3NC42NDQzIDQ0My4yNSBMIDQyOC42NDQzIDQ5NSBMIDQyOC42NDQzIDMzOS43NSBaIiBzdHJva2U9ImJsYWNrIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0cm9rZS13aWR0aD0iMSIvPjxyZWN0IHg9IjQyOC42NDQzIiB5PSIzMzkuNzUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxNTUuMjUiIGZpbGw9InVybCgjT2JqX0dyYWRpZW50XzIpIi8+PHJlY3QgeD0iNDI4LjY0NDMiIHk9IjMzOS43NSIgd2lkdGg9IjE4IiBoZWlnaHQ9IjE1NS4yNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48cGF0aCBkPSJNIDM3NC42NDQzIDI4OCBMIDQyOC42NDQzIDMzOS43NSBMIDQ0Ni42NDQzIDMzOS43NSBMIDM5Mi42NDQzIDI4OCBaIiBmaWxsPSJ1cmwoI09ial9HcmFkaWVudF8zKSIvPjxwYXRoIGQ9Ik0gMzc0LjY0NDMgMjg4IEwgNDI4LjY0NDMgMzM5Ljc1IEwgNDQ2LjY0NDMgMzM5Ljc1IEwgMzkyLjY0NDMgMjg4IFoiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxIi8+PHBhdGggZD0iTSA3MDIgMzg5LjY4NDU2IEwgNzAyIDQ1Ny4xODQ1NiBMIDczNS43NSA0NzkuNjg0NTYgTCA3MzUuNzUgNDEyLjE4NDU2IFoiIGZpbGw9InVybCgjT2JqX0dyYWRpZW50XzQpIi8+PHBhdGggZD0iTSA3MDIgMzg5LjY4NDU2IEwgNzAyIDQ1Ny4xODQ1NiBMIDczNS43NSA0NzkuNjg0NTYgTCA3MzUuNzUgNDEyLjE4NDU2IFoiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxIi8+PHJlY3QgeD0iNzM1Ljc1IiB5PSI0MTIuMTg0NTYiIHdpZHRoPSIxMS4yNSIgaGVpZ2h0PSI2Ny41IiBmaWxsPSJ1cmwoI09ial9HcmFkaWVudF81KSIvPjxyZWN0IHg9IjczNS43NSIgeT0iNDEyLjE4NDU2IiB3aWR0aD0iMTEuMjUiIGhlaWdodD0iNjcuNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48cGF0aCBkPSJNIDcwMiAzODkuNjg0NTYgTCA3MzUuNzUgNDEyLjE4NDU2IEwgNzQ3IDQxMi4xODQ1NiBMIDcxNC44NTcxNCAzODkuNjg0NTYgWiIgZmlsbD0idXJsKCNPYmpfR3JhZGllbnRfNikiLz48cGF0aCBkPSJNIDcwMiAzODkuNjg0NTYgTCA3MzUuNzUgNDEyLjE4NDU2IEwgNzQ3IDQxMi4xODQ1NiBMIDcxNC44NTcxNCAzODkuNjg0NTYgWiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48bGluZSB4MT0iNDQ2LjY0NDMiIHkxPSI0MTcuNDI1MTUiIHgyPSI1MjQuMTAwMSIgeTI9IjQxNy44NTY3NSIgbWFya2VyLWVuZD0idXJsKCNTaGFycEFycm93X01hcmtlcikiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxIi8+PHRleHQgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDY5Ljc4ODYgMzkzLjc2MjU4KSIgZmlsbD0iYmxhY2siPjx0c3BhbiBmb250LWZhbWlseT0iSGVsdmV0aWNhIE5ldWUiIGZvbnQtc2l6ZT0iMTYiIGZvbnQtd2VpZ2h0PSI1MDAiIHg9Ii40MiIgeT0iMTUiIHRleHRMZW5ndGg9IjM2LjE2Ij5ib2R5PC90c3Bhbj48L3RleHQ+PGxpbmUgeDE9IjU4NC4zMjg4NiIgeTE9IjM3Ny42ODQ1NiIgeDI9IjY1OS4xMDAwNSIgeTI9IjM3Ny45NzMzNSIgbWFya2VyLWVuZD0idXJsKCNTaGFycEFycm93X01hcmtlcikiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxIi8+PGxpbmUgeDE9IjU0OSIgeTE9IjM1MS4zMTU0NCIgeDI9IjYyMy43NzEyIiB5Mj0iMzUxLjAyNjY1IiBtYXJrZXItZW5kPSJ1cmwoI1NoYXJwQXJyb3dfTWFya2VyKSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48bGluZSB4MT0iNTk0IiB5MT0iNDM1Ljk0NzIyIiB4Mj0iNjk1LjEwMDM1IiB5Mj0iNDM0LjkyNDk2IiBtYXJrZXItZW5kPSJ1cmwoI1NoYXJwQXJyb3dfTWFya2VyKSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48dGV4dCB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1NTkuNTYyNSA1MTIuNzc2KSIgZmlsbD0iYmxhY2siPjx0c3BhbiBmb250LWZhbWlseT0iSGVsdmV0aWNhIE5ldWUiIGZvbnQtc2l6ZT0iMTYiIGZvbnQtd2VpZ2h0PSI1MDAiIHg9Ii4xNjQiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI1MC42NzIiPnNjYWxlIDA8L3RzcGFuPjwvdGV4dD48dGV4dCB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3MTggNTEyLjE1NTEzKSIgZmlsbD0iYmxhY2siPjx0c3BhbiBmb250LWZhbWlseT0iSGVsdmV0aWNhIE5ldWUiIGZvbnQtc2l6ZT0iMTYiIGZvbnQtd2VpZ2h0PSI1MDAiIHg9Ii4xNjQiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI1MC42NzIiPnNjYWxlIDE8L3RzcGFuPjwvdGV4dD48dGV4dCB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0MTcuMDYyNSA1MTIuNzc2KSIgZmlsbD0iYmxhY2siPjx0c3BhbiBmb250LWZhbWlseT0iSGVsdmV0aWNhIE5ldWUiIGZvbnQtc2l6ZT0iMTYiIGZvbnQtd2VpZ2h0PSI1MDAiIHg9Ii4wNjQiIHk9IjE1IiB0ZXh0TGVuZ3RoPSIzNS44NzIiPmlucHV0PC90c3Bhbj48L3RleHQ+PHRleHQgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNjAxLjM5MTM2IDQxMS4wOTE0NCkiIGZpbGw9ImJsYWNrIj48dHNwYW4gZm9udC1mYW1pbHk9IkhlbHZldGljYSBOZXVlIiBmb250LXNpemU9IjE2IiBmb250LXdlaWdodD0iNTAwIiB4PSIuMjE2IiB5PSIxNSIgdGV4dExlbmd0aD0iOTEuNTY4Ij5kb3duc2FtcGxlPC90c3Bhbj48L3RleHQ+PHRleHQgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNTU2LjE5OCAzMjcuMTMxNykiIGZpbGw9ImJsYWNrIj48dHNwYW4gZm9udC1mYW1pbHk9IkhlbHZldGljYSBOZXVlIiBmb250LXNpemU9IjE2IiBmb250LXdlaWdodD0iNTAwIiB4PSIuMjU2IiB5PSIxNSIgdGV4dExlbmd0aD0iNTYiPmNsYXNzIHByPC90c3Bhbj48dHNwYW4gZm9udC1mYW1pbHk9IkhlbHZldGljYSBOZXVlIiBmb250LXNpemU9IjE2IiBmb250LXdlaWdodD0iNTAwIiB4PSI1NS45NjgiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI0OS43NzYiPmVkaWN0b3I8L3RzcGFuPjwvdGV4dD48dGV4dCB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDIgMzU0LjgwMjg1KSIgZmlsbD0iYmxhY2siPjx0c3BhbiBmb250LWZhbWlseT0iSGVsdmV0aWNhIE5ldWUiIGZvbnQtc2l6ZT0iMTYiIGZvbnQtd2VpZ2h0PSI1MDAiIHg9Ii4xNDQiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI0Ni4yMjQiPmJveCBwcjwvdHNwYW4+PHRzcGFuIGZvbnQtZmFtaWx5PSJIZWx2ZXRpY2EgTmV1ZSIgZm9udC1zaXplPSIxNiIgZm9udC13ZWlnaHQ9IjUwMCIgeD0iNDYuMDgiIHk9IjE1IiB0ZXh0TGVuZ3RoPSI0OS43NzYiPmVkaWN0b3I8L3RzcGFuPjwvdGV4dD48cGF0aCBkPSJNIDUzMSAzNDYuMDI2ODUgTCA1MzEgNDU0LjAyNjg1IEwgNTc4LjI1IDQ5MC4wMjY4NSBMIDU3OC4yNSAzODIuMDI2ODUgWiIgZmlsbD0idXJsKCNPYmpfR3JhZGllbnRfNykiLz48cGF0aCBkPSJNIDUzMSAzNDYuMDI2ODUgTCA1MzEgNDU0LjAyNjg1IEwgNTc4LjI1IDQ5MC4wMjY4NSBMIDU3OC4yNSAzODIuMDI2ODUgWiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48cmVjdCB4PSI1NzguMjUiIHk9IjM4Mi4wMjY4NSIgd2lkdGg9IjE1Ljc1IiBoZWlnaHQ9IjEwOCIgZmlsbD0idXJsKCNPYmpfR3JhZGllbnRfOCkiLz48cmVjdCB4PSI1NzguMjUiIHk9IjM4Mi4wMjY4NSIgd2lkdGg9IjE1Ljc1IiBoZWlnaHQ9IjEwOCIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48cGF0aCBkPSJNIDUzMSAzNDYuMDI2ODUgTCA1NzguMjUgMzgyLjAyNjg1IEwgNTk0IDM4Mi4wMjY4NSBMIDU0OSAzNDYuMDI2ODUgWiIgZmlsbD0idXJsKCNPYmpfR3JhZGllbnRfOSkiLz48cGF0aCBkPSJNIDUzMSAzNDYuMDI2ODUgTCA1NzguMjUgMzgyLjAyNjg1IEwgNTk0IDM4Mi4wMjY4NSBMIDU0OSAzNDYuMDI2ODUgWiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2Utd2lkdGg9IjEiLz48L2c+PC9nPjwvc3ZnPg==)

We first use a `body` network to extract the image features,
which are used as the input to the first scale (scale 0). The class labels and the corresponding anchor boxes
are predicted by `class_predictor` and `box_predictor`, respectively.
We then downsample the representations to the next scale (scale 1).
Again, at this new resolution, we predict both classes and anchor boxes.
This downsampling and predicting routine
can be repeated in multiple times to obtain results on multiple resolution scales.
Let's walk through the components one by one in a bit more detail.

### Default anchor boxes

Since an anchor box can have arbituary shape,
we sample a set of anchor boxes as the candidate.
In particular, for each pixel, we sample multiple boxes
centered at this pixel but have various sizes and ratios.
Assume the input size is $w \times h$,
- for size $s\in (0,1]$, the generated box shape will be $ws \times hs$
- for ratio $r > 0$, the generated box shape will be $w\sqrt{r} \times \frac{h}{\sqrt{r}}$

We can sample the boxes using the operator `MultiBoxPrior`.
It accepts *n* sizes and *m* ratios to generate *n+m-1* boxes for each pixel.
The first *i* boxes are generated from `sizes[i], ratios[0]`
if $i \le n$ otherwise `sizes[0], ratios[i-n]`.

```{.python .input  n=1}
import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior

n = 40
# shape: batch x channel x height x weight
x = nd.random_uniform(shape=(1, 3, n, n))

y = MultiBoxPrior(x, sizes=[.5, .25, .1], ratios=[1, 2, .5])

# the first anchor box generated for pixel at (20,20)
# its format is (x_min, y_min, x_max, y_max)
boxes = y.reshape((n, n, -1, 4))
print('The first anchor box at row 21, column 21:', boxes[20, 20, 0, :])
```

We can visualize all anchor boxes generated for one pixel on a certain size feature map.

```{.python .input  n=2}
import matplotlib.pyplot as plt
def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]),
        fill=False, edgecolor=color, linewidth=linewidth)
colors = ['blue', 'green', 'red', 'black', 'magenta']
plt.imshow(nd.ones((n, n, 3)).asnumpy())
anchors = boxes[20, 20, :, :]
for i in range(anchors.shape[0]):
    plt.gca().add_patch(box_to_rect(anchors[i,:]*n, colors[i]))
plt.show()
```

### Predict classes

For each anchor box, we want to predict the associated class label.
We make this prediction by using a convolution layer.
We choose a kernel of size $3\times 3$ with padding size $(1, 1)$
so that the output will have the same width and height as the input.
The confidence scores for the anchor box class labels are stored in channels.
In particular, for the *i*-th anchor box:

- channel `i*(num_class+1)` store the scores for this box contains only background
- channel `i*(num_class+1)+1+j` store the scores for this box contains an object from the *j*-th class

```{.python .input  n=3}
from mxnet.gluon import nn
def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

cls_pred = class_predictor(5, 10)
cls_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
print('Class prediction', cls_pred(x).shape)
```

### Predict anchor boxes

The goal is predict how to transfer the current anchor box to the correct box. That is, assume $b$ is one of the sampled default box, while $Y$ is the ground truth, then we want to predict the delta positions $\Delta(Y, b)$, which is a 4-length vector.

More specifically, the we define the delta vector as:
[$t_x$, $t_y$, $t_{width}$, $t_{height}$], where

- $t_x = (Y_x - b_x) / b_{width}$
- $t_y = (Y_y - b_y) / b_{height}$
- $t_{width} = (Y_{width} - b_{width}) / b_{width}$
- $t_{height} = (Y_{height} - b_{height}) / b_{height}$

Normalizing the deltas with box width/height tends to result in better convergence behavior.

Similar to classes, we use a convolution layer here. The only difference is that the output channel size is now `num_anchors * 4`, with the predicted delta positions for the *i*-th box stored from channel `i*4` to `i*4+3`.

```{.python .input  n=4}
def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

box_pred = box_predictor(10)
box_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
print('Box prediction', box_pred(x).shape)
```

### Down-sample features

Each time, we downsample the features by half.
This can be achieved by a simple pooling layer with pooling size 2.
We may also stack two convolution, batch normalization and ReLU blocks
before the pooling layer to make the network deeper.

```{.python .input  n=5}
def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

blk = down_sample(10)
blk.initialize()
x = nd.zeros((2, 3, 20, 20))
print('Before', x.shape, 'after', blk(x).shape)
```

### Manage preditions from multiple layers

A key property of SSD is that predictions are made
at multiple layers with shrinking spatial size.
Thus, we have to handle predictions from multiple feature layers.
One idea is to concatenate them along convolutional channels,
with each one predicting a correspoding value(class or box) for each default anchor.
We give class predictor as an example, and box predictor follows the same rule.

```{.python .input  n=6}
# a certain feature map with 20x20 spatial shape
feat1 = nd.zeros((2, 8, 20, 20))
print('Feature map 1', feat1.shape)
cls_pred1 = class_predictor(5, 10)
cls_pred1.initialize()
y1 = cls_pred1(feat1)
print('Class prediction for feature map 1', y1.shape)
# down-sample
ds = down_sample(16)
ds.initialize()
feat2 = ds(feat1)
print('Feature map 2', feat2.shape)
cls_pred2 = class_predictor(3, 10)
cls_pred2.initialize()
y2 = cls_pred2(feat2)
print('Class prediction for feature map 2', y2.shape)
```

```{.python .input  n=7}
def flatten_prediction(pred):
    return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

def concat_predictions(preds):
    return nd.concat(*preds, dim=1)

flat_y1 = flatten_prediction(y1)
print('Flatten class prediction 1', flat_y1.shape)
flat_y2 = flatten_prediction(y2)
print('Flatten class prediction 2', flat_y2.shape)
print('Concat class predictions', concat_predictions([flat_y1, flat_y2]).shape)
```

### Body network

The body network is used to extract features from the raw pixel inputs. Common choices follow the architectures of the state-of-the-art convolution neural networks for image classification. For demonstration purpose, we just stack several down sampling blocks to form the body network.

```{.python .input  n=8}
from mxnet import gluon
def body():
    """return the body network"""
    out = nn.HybridSequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(nfilters))
    return out

bnet = body()
bnet.initialize()
x = nd.zeros((2, 3, 256, 256))
print('Body network', [y.shape for y in bnet(x)])
```

### Create a toy SSD model

Now, let's create a toy SSD model that takes images of resolution $256 \times 256$ as input.

```{.python .input  n=9}
def toy_ssd_model(num_anchors, num_classes):
    """return SSD modules"""
    downsamples = nn.Sequential()
    class_preds = nn.Sequential()
    box_preds = nn.Sequential()

    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))

    for scale in range(5):
        class_preds.add(class_predictor(num_anchors, num_classes))
        box_preds.add(box_predictor(num_anchors))

    return body(), downsamples, class_preds, box_preds

print(toy_ssd_model(5, 2))
```

### Forward

Given an input and the model, we can run the forward pass.

```{.python .input  n=10}
def toy_ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):
    # extract feature with the body network
    x = body(x)

    # for each scale, add anchors, box and class predictions,
    # then compute the input to next scale
    default_anchors = []
    predicted_boxes = []
    predicted_classes = []

    for i in range(5):
        default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        predicted_boxes.append(flatten_prediction(box_preds[i](x)))
        predicted_classes.append(flatten_prediction(class_preds[i](x)))
        if i < 3:
            x = downsamples[i](x)
        elif i == 3:
            # simply use the pooling layer
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))

    return default_anchors, predicted_classes, predicted_boxes
```

### Put all things together

```{.python .input  n=11}
from mxnet import gluon
class ToySSD(gluon.Block):
    def __init__(self, num_classes, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor box sizes for 4 feature scales
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # anchor box ratios for 4 feature scales
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = toy_ssd_model(4, num_classes)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = toy_ssd_forward(x, self.body, self.downsamples,
            self.class_preds, self.box_preds, self.anchor_sizes, self.anchor_ratios)
        # we want to concatenate anchors, class predictions, box predictions from different layers
        anchors = concat_predictions(default_anchors)
        box_preds = concat_predictions(predicted_boxes)
        class_preds = concat_predictions(predicted_classes)
        # it is better to have class predictions reshaped for softmax computation
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))

        return anchors, class_preds, box_preds
```

### Outputs of ToySSD

```{.python .input  n=12}
# instantiate a ToySSD network with 10 classes
net = ToySSD(2)
net.initialize()
x = nd.zeros((1, 3, 256, 256))
default_anchors, class_predictions, box_predictions = net(x)
print('Outputs:', 'anchors', default_anchors.shape, 'class prediction', class_predictions.shape, 'box prediction', box_predictions.shape)
```

## Dataset

For demonstration purposes, we'll build a train our model to detect Pikachu in the wild.
We generated a a synthetic toy dataset by rendering images from open-sourced 3D Pikachu models.
The dataset consists of 1000 pikachus with random pose/scale/position in random background images.
The exact locations are recorded as ground-truth for training and validation.

![](https://user-images.githubusercontent.com/3307514/29479494-5dc28a02-8427-11e7-91d0-2849b88c17cd.png)


### Download dataset

```{.python .input  n=13}
from mxnet.test_utils import download
import os.path as osp
def verified(file_path, sha1hash):
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    matched = sha1.hexdigest() == sha1hash
    if not matched:
        print('Found hash mismatch in file {}, possibly due to incomplete download.'.format(file_path))
    return matched

url_format = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/{}'
hashes = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
          'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
          'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
for k, v in hashes.items():
    fname = 'pikachu_' + k
    target = osp.join('data', fname)
    url = url_format.format(k)
    if not osp.exists(target) or not verified(target, v):
        print('Downloading', target, url)
        download(url, fname=fname, dirname='data', overwrite=True)
```

### Load dataset

```{.python .input  n=14}
import mxnet.image as image
data_shape = 256
batch_size = 32
def get_iterators(data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_train.rec',
        path_imgidx='./data/pikachu_train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class

train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
batch = train_data.next()
print(batch)
```

### Illustration

Let's display one image loaded by ImageDetIter.

```{.python .input  n=15}
import numpy as np

img = batch.data[0][0].asnumpy()  # grab the first image, convert to numpy array
img = img.transpose((1, 2, 0))  # we want channel to be the last dimension
img += np.array([123, 117, 104])
img = img.astype(np.uint8)  # use uint8 (0-255)
# draw bounding boxes on image
for label in batch.label[0][0].asnumpy():
    if label[0] < 0:
        break
    print(label)
    xmin, ymin, xmax, ymax = [int(x * data_shape) for x in label[1:5]]
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=(1, 0, 0), linewidth=3)
    plt.gca().add_patch(rect)
plt.imshow(img)
plt.show()
```

## Train

### Losses

Network predictions will be penalized for incorrect class predictions and wrong box deltas.

```{.python .input  n=16}
from mxnet.contrib.ndarray import MultiBoxTarget
def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
    z = MultiBoxTarget(*[default_anchors, labels, class_predicts])
    box_target = z[0]  # box offset target for (x, y, width, height)
    box_mask = z[1]  # mask is used to ignore box offsets we don't want to penalize, e.g. negative samples
    cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
    return box_target, box_mask, cls_target
```

Pre-defined losses are provided in `gluon.loss` package, however, we can define losses manually.

First, we need a Focal Loss for class predictions.

```{.python .input  n=17}
class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pt = F.pick(output, label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

# cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
cls_loss = FocalLoss()
print(cls_loss)
```

Next, we need a SmoothL1Loss for box predictions.

```{.python .input  n=18}
class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)

box_loss = SmoothL1Loss()
print(box_loss)
```

### Evaluation metrics

Here, we define two metrics that we'll use to evaluate our performance whien training.
You're already familiar with accuracy unless you've been naughty and skipped straight to object detection.
We use the accuracy metric to assess the quality of the class predictions.
Mean absolute error (MAE) is just the L1 distance, introduced in our [linear algebra chapter](../chapter01_crashcourse/linear-algebra.ipynb).
We use this to determine how close the coordinates of the predicted bounding boxes are to the ground-truth coordinates.
Because we are jointly solving both a classification problem and a regression problem, we need an appropriate metric for each task.

```{.python .input  n=19}
cls_metric = mx.metric.Accuracy()
box_metric = mx.metric.MAE()  # measure absolute difference between prediction and target
```

```{.python .input  n=20}
### Set context for training
ctx = mx.gpu()  # it may takes too long to train using CPU
try:
    _ = nd.zeros(1, ctx=ctx)
    # pad label for cuda implementation
    train_data.reshape(label_shape=(3, 5))
    train_data = test_data.sync_label_shape(train_data)
except mx.base.MXNetError as err:
    print('No GPU enabled, fall back to CPU, sit back and be patient...')
    ctx = mx.cpu()
```

### Initialize parameters

```{.python .input  n=21}
net = ToySSD(num_class)
net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
```

### Set up trainer

```{.python .input  n=22}
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})
```

### Start training

Optionally we load pretrained model for demonstration purpose. One can set `from_scratch = True` to training from scratch, which may take more than 30 mins to finish using a single capable GPU.

```{.python .input  n=23}
epochs = 150  # set larger to get better performance
log_interval = 20
from_scratch = False  # set to True to train from scratch
if from_scratch:
    start_epoch = 0
else:
    start_epoch = 148
    pretrained = 'ssd_pretrained.params'
    sha1 = 'fbb7d872d76355fff1790d864c2238decdb452bc'
    url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/ssd_pikachu-fbb7d872.params'
    if not osp.exists(pretrained) or not verified(pretrained, sha1):
        print('Downloading', pretrained, url)
        download(url, fname=pretrained, overwrite=True)
    net.load_params(pretrained, ctx)
```

```{.python .input  n=24}
import time
from mxnet import autograd as ag
for epoch in range(start_epoch, epochs):
    # reset iterator and tick
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    # iterate through all batch
    for i, batch in enumerate(train_data):
        btic = time.time()
        # record gradients
        with ag.record():
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            default_anchors, class_predictions, box_predictions = net(x)
            box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
            # losses
            loss1 = cls_loss(class_predictions, cls_target)
            loss2 = box_loss(box_predictions, box_target, box_mask)
            # sum all losses
            loss = loss1 + loss2
            # backpropagate
            loss.backward()
        # apply
        trainer.step(batch_size)
        # update metrics
        cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
        box_metric.update([box_target], [box_predictions * box_mask])
        if (i + 1) % log_interval == 0:
            name1, val1 = cls_metric.get()
            name2, val2 = box_metric.get()
            print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
                  %(epoch ,i, batch_size/(time.time()-btic), name1, val1, name2, val2))

    # end of epoch logging
    name1, val1 = cls_metric.get()
    name2, val2 = box_metric.get()
    print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

# we can save the trained parameters to disk
net.save_params('ssd_%d.params' % epochs)
```

## Test

Testing is similar to training, except that we don't need to compute gradients and training targets. Instead, we take the predictions from network output, and combine them to get the real detection output.

### Prepare the test data

```{.python .input  n=25}
import numpy as np
def preprocess(im):
    """Takes an image and apply preprocess"""
    # resize to data_shape
    im = image.imresize(im, data_shape, data_shape)
    # swap BGR to RGB
    # im = im[:, :, (2, 1, 0)]
    # convert to float before subtracting mean
    im = im.astype('float32')
    # subtract mean
    im -= nd.array([123, 117, 104])
    # organize as [batch-channel-height-width]
    im = im.transpose((2, 0, 1))
    im = im.expand_dims(axis=0)
    return im

with open('../img/pikachu.jpg', 'rb') as f:
    im = image.imdecode(f.read())
x = preprocess(im)
print('x', x.shape)
```

### Network inference

In a single line of code!

```{.python .input  n=26}
# if pre-trained model is provided, we can load it
# net.load_params('ssd_%d.params' % epochs, ctx)
anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
print('anchors', anchors)
print('class predictions', cls_preds)
print('box delta predictions', box_preds)
```

### Convert predictions to real object detection results

```{.python .input  n=27}
from mxnet.contrib.ndarray import MultiBoxDetection
# convert predictions to probabilities using softmax
cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
# apply shifts to anchors boxes, non-maximum-suppression, etc...
output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress=True, clip=False)
print(output)
```

Each row in the output corresponds to a detection box, as in format [class_id, confidence, xmin, ymin, xmax, ymax].

Most of the detection results are -1, indicating that they either have very small confidence scores, or been suppressed through non-maximum-suppression.

### Display results

```{.python .input  n=28}
def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img.asnumpy())
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                             edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()

display(im, output[0].asnumpy(), thresh=0.45)
```

## Conclusion

Detection is harder than classification, since we want not only class probabilities, but also localizations of different objects including potential small objects. Using sliding window together with a good classifier might be an option, however, we have shown that with a properly designed convolutional neural network, we can do single shot detection which is blazing fast and accurate!

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
