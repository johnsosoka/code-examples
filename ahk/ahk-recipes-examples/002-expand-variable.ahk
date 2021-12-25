F1::
{
    name_variable := "john"
    Msgbox, hello there %name_variable%
    return
}

; Using Format
F2::
{
    name_variable := "john3"
    MsgBox, % Format("hello there {}", name_variable)
}
