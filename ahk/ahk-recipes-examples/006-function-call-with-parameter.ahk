F1::
{
    Msgbox, % get_hello_message_function("john")
    return
}

get_hello_message_function(name)
{
    ;Msgbox, %name%
    message := Format("hello {}", name)
    return message
}