#Requires AutoHotkey v2.0
#SingleInstance Force

logPath := A_ScriptDir "\neutron_log.txt"
SendMode("Event")  ; More compatible with Elite Dangerous

F11::{
    SetKeyDelay(30, 30)
    Log("Hotkey triggered (F11)")

    Sleep(500)
    Send("1")
    Log("Sent 1")

    Sleep(500)
    Send("a")
    Log("Sent a")

    Sleep(500)
    Send("{Space}")
    Log("Sent Space")

    Sleep(3000)
    Send("{Left}")
    Log("Sent Left Arrow")

    Sleep(500)
    Send("{Right}")
    Log("Sent Right Arrow")

    Sleep(1500)
    Send("{Space}")
    Log("Sent Space again")

    Sleep(500)
    Send("^v")  ; Ctrl+V to paste
    Log("Pasted (Ctrl+V)")

    Sleep(500)
    Send("{Down}")
    Log("Sent Down Arrow")

    Sleep(1500)
    Send("{Space}")
    Log("Sent Space")

    ; Move mouse to center of the screen and wiggle a little
    centerX := A_ScreenWidth // 2
    centerY := A_ScreenHeight // 2
    MouseMove(centerX - 10, centerY - 10, 10)
    Sleep(100)
    MouseMove(centerX + 10, centerY + 10, 10)
    Sleep(100)
    MouseMove(centerX, centerY, 10)
    Log("Mouse moved to center and wiggled")

    ; Press and hold left mouse button
    Click("L", "D")  ; Press down
    Log("Mouse left button down")
    Sleep(1500)
    Click("L", "U")  ; Release
    Log("Mouse left button up after 1.5s")

    ; Wait and press ESC twice
    Sleep(500)
    Send("{Esc}")
    Log("Sent ESC")
    Sleep(200)
    Send("{Esc}")
    Log("Sent ESC again")
}

Log(msg) {
    global logPath
    FileAppend(Format("[{1:yyyy-MM-dd HH:mm:ss}] {2}`n", A_Now, msg), logPath)
}
