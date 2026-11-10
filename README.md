# Hyeongjin Kim · Personal Academic Homepage

경제학 대학원생 **Hyeongjin Kim**의 개인 연구, 강의, 프로젝트 및 스터디 노트를 담은 Quarto 기반의 현대적인 아카데믹 웹사이트입니다.

🌐 **웹사이트 주소:** [https://hyeongjin3544.github.io/HyeongjinHome/](https://hyeongjin3544.github.io/HyeongjinHome/)

---

## 📁 사이트 구조

```text
├── index.qmd               # 홈 & 프로필 (Hero, 연구 하이라이트, 최근 노트, 소식)
├── research.qmd            # 연구 (Working Papers, Works in Progress, Publications)
├── teaching.qmd            # 강의 및 조교 경력 (TA, 실습 랩, 강의계획서)
├── projects.qmd            # 데이터 분석 & 코드 포트폴리오 (R, Python, Stata)
├── notes.qmd               # 공부 노트 및 페이퍼 리뷰 아카이브 (자동 필터링 목록)
├── cv.qmd                  # 상세 학술 이력서 (CV)
├── about.qmd               # 상세 소개 및 오피스/연락처
├── notes/                  # 개별 경제학 노트 및 포스트 (.qmd)
│   ├── causal-inference-did.qmd
│   └── linear-market.qmd
├── assets/                 # 프로필 아바타, 다이어그램, 이미지
├── templates/              # 새 글 작성용 템플릿
│   └── note.qmd
├── styles.scss             # 모던 아카데믹 맞춤 SCSS 스타일
├── _quarto.yml             # Quarto 전역 웹사이트 설정
└── .github/workflows/      # GitHub Actions 자동 빌드 & 배포 워크플로우
```

---

## ✏️ 내 정보로 커스터마이징하는 방법

1. **프로필 및 기본 정보 변경:**
   - `index.qmd`, `about.qmd`, `cv.qmd`에서 본인의 영문/국문 성명, 소속 학교/학과, 이메일 주소를 변경합니다.
   - `assets/avatar.svg`를 본인의 실제 프로필 사진(예: `assets/profile.jpg`)으로 교체하고 파일 경로를 수정할 수 있습니다.
2. **연구 및 워킹페이퍼 추가:**
   - `research.qmd`에서 본인의 논문 제목, 초록, PDF 링크를 수정합니다.
3. **새로운 공부 노트 작성:**
   - `templates/note.qmd`를 복사하여 `notes/` 폴더에 새 파일(예: `notes/my-new-post.qmd`)로 저장합니다.
   - 상단의 `date`와 `categories`를 지정하면 `notes.qmd` 목록에 자동으로 정렬되어 표시됩니다.
4. **이력서(CV) PDF 업로드:**
   - 본인의 CV PDF 파일을 `assets/cv.pdf`로 저장하고 `cv.qmd`의 다운로드 링크에 연결합니다.

---

## 🚀 자동 배포 (GitHub Pages)

이 저장소는 GitHub Actions(`.github/workflows/publish.yml`)가 설정되어 있어, 코드를 `main` 브랜치에 `git push`하기만 하면 자동으로 최신 웹사이트로 빌드되어 배포됩니다.

> **최초 설정 확인:**
> GitHub 저장소 페이지의 **Settings** → **Pages** → **Build and deployment** → **Source**가 **GitHub Actions**로 설정되어 있는지 확인하세요.
