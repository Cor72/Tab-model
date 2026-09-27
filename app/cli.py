from app.model import CompletionModel


def main() -> None:
    print("Local Completion v0.1")
    model = CompletionModel()

    while True:
        prefix = input("\n输入（直接回车退出）：")
        if not prefix:
            break

        print("补全：", model.complete(prefix))


if __name__ == "__main__":
    main()