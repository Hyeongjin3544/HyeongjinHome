# Hyeongjin의 개인 홈페이지

경제학 대학원생의 개인 소개와 공부 노트를 담은 Quarto 웹사이트입니다.

## 내용 수정

- `index.qmd`에서 첫 화면의 소개 문구를 바꿉니다.
- `about.qmd`에서 소개와 공개 프로필 링크를 관리합니다.
- 새 노트는 `templates/note.qmd`를 복사해 `notes/` 폴더에 저장합니다. 날짜와 분야를 채우면 경제학 노트 목록에 자동으로 나타납니다.

## 미리보기와 배포

Quarto가 설치되어 있으면 이 폴더에서 `quarto preview`로 미리 보고, `quarto render`로 전체 사이트를 생성할 수 있습니다. RStudio에 포함된 Quarto를 사용한다면 RStudio 터미널에서도 이 명령을 실행할 수 있습니다.

GitHub 저장소의 **Settings → Pages → Build and deployment → Source**에서 **GitHub Actions**를 선택합니다. 이후 `main`에 변경사항을 올리면 자동으로 배포합니다. 첫 배포가 자동으로 시작되지 않으면 **Actions → Publish homepage → Run workflow**에서 직접 실행합니다.

배포 주소는 [Hyeongjin의 개인 홈페이지](https://hyeongjin3544.github.io/HyeongjinHome/)입니다.
