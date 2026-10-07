use std::sync::{atomic::{AtomicBool, Ordering},Mutex}
use tauri::{AppHandle, Manager, Runtime, Window,Emitter,Manager};
use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

struct PanelState {
    status:Mutex<string>,
}

#[tauri::command]
fn get_status(state: tauri::State<'_, PanelState>) -> Result<String, String> {
    state.status.lock().map(|status| status.clone()).map_err(|_| "Failed to lock status".to_string())
}

fn update_status(app:&tauri::AppHandle,massage:&str){
    let state = app.state::<PanelState>();
    match state.status.lock() {
        Ok(mut status) => {
            *status = massage.to_string();
        }
        Err(_) => {
            println!("Failed to lock status");
        }
    }
    if let Err(error) = app.emit("completion-state-changed", message) {
        eprintln!("发送状态事件失败：{error}");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(PanelState {
            status: Mutex::new("正在注册快捷键……".to_string()),
        })
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let shortcuts = [
                ("Ctrl+Alt+Space", "收到：触发补全"),
                ("Ctrl+Alt+Enter", "收到：接受补全"),
                ("Ctrl+Alt+Backspace", "收到：取消补全"),
            ];

            for (shortcut, message) in shortcuts {
                // 每个快捷键各自记录按下状态，避免长按重复触发。
                let held = AtomicBool::new(false);

                let result = app.global_shortcut().on_shortcut(
                    shortcut,
                    move |app, _shortcut, event| match event.state {
                        ShortcutState::Pressed => {
                            if !held.swap(true, Ordering::Relaxed) {
                                update_status(app, message);
                            }
                        }
                        ShortcutState::Released => {
                            held.store(false, Ordering::Relaxed);
                        }
                    },
                );

                if let Err(error) = result {
                    // 一个注册失败，就释放已经注册的其他快捷键。
                    if let Err(cleanup_error) =
                        app.global_shortcut().unregister_all()
                    {
                        eprintln!("释放快捷键失败：{cleanup_error}");
                    }

                    update_status(
                        app.handle(),
                        &format!(
                            "快捷键注册失败：{shortcut}。\
                             请检查是否被其他程序占用。详情：{error}"
                        ),
                    );

                    return Ok(());
                }
            }

            update_status(app.handle(), "三个快捷键已注册，等待操作");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_status])
        .run(tauri::generate_context!())
        .expect("启动 Kagami 失败");
}
