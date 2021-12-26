F2::
{
    sleep_duration(2000, 5000)
    Msgbox "finished sleeping"
}

sleep_duration(min_milliseconds, max_milliseconds)
{
    Random, sleepDurationAmount, %min_milliseconds%, %max_milliseconds%
    sleep sleepDurationAmount
}