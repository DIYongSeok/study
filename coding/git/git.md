# github

## 코드 업로드 과정
1) 프로젝터 폴더 git 설정 – git init
2) 변경한 파일 중 올리길 원하는 것 선택 – git add
3) 선택한 파일들을 한 덩어리로 만들고 설명 첨삭 – git commit –m “description”
4) github 사이트 프로젝트 저장소(repository) 만들기
5) github과 프로젝터 폴더 연결 – git remote add
6) 프로젝트 폴더 repository에 업로드 – git push

## 명령어

> git init
- git init을 입력하여 초기화
- .git 폴더가 생성되며, 버전 정보, 원격 저장소 주소 등이 저장된다.
- 원격 저장소에서 컴퓨터로 코드를 받아오면 로컬 저장소가 자동으로 생성
- 한 폴더에는 하나의 .git만 존재해야 한다.

> git add [file name]
- 커밋으로 만들길 원하는 파일 선택

> git commit –m “message”
- 현재 시점에서 파일들의 상태를 묶어둔 덩어리

> git log
- 커밋된 것들을 확인 가능

> git remote add [repository nickname] [link]
- 원격 저장소에 컴퓨터의 로컬 저장소를 연결함

> git push [repository nickname] [branch]
- 원격 저장소의 해당 branch에 커밋한 것들 업로드

> git clone [link]
- 원격 저장소로부터 코드를 컴퓨터의 현재 directory로 import 함

> git pull [repository nickname] [branch] = fetch+merge
- 원격 저장소의 지정된 branch에 업데이트된 코드를 컴퓨터의 현재 branch로 받아옴

---

> __branch__
- 두 명 이상의 작업자가 서로 다른 충돌되는 코드를 만들 경우, 브랜치(가지)를 만들어 코드를 생산하면 됨
- HEAD: 내가 지금 작업하고 있는 로컬 브랜치를 가리킴

---

## 명령어
> git branch [branch name]
- 새로운 branch를 생성
- git branch 만 입력하면, branch 목록 출력

> git checkout [branch name]
- 해당 브랜치로 HEAD 이동

> git merge [branch name] - 두 브랜치의 합집합
- 현재의 branch에 다른 branch 병합
- 컨플릭트 발생 처리

![image](https://github.com/user-attachments/assets/55dc5171-eb2a-404e-a8bf-5fc01a01369a)



---
## 소규모 팀에서의 git 활용


> git fork & git pull request (깃헙 사이트에서 진행됨)
- 제 3자가 자신의 원격 저장소에 타 그룹의 repository를 통째로 복사한 후, 자신의 branch를 commit&push 하고 해당 내용을 다시 타 그룹의 repository로 merge하기 위해 pull request로 요청을 보냄

![image](https://github.com/user-attachments/assets/059b5490-e2ab-441b-aaf5-98425f8d17db)

> 추가명령어
1. **amend**: 깜빡하고 수정 못 한 파일이 있어요, 방금 만든 커밋에 살짝 추가할래요
2. **cherry-pick**: 저 커밋 하나만 떼서 지금 브랜치에 붙이고 싶어요
3. **reset**: 옛날 커밋으로 시간을 돌리고 싶어요
4. **reverse**: 이 커밋의 변경사항을 되돌리고 싶어요
5. **stash**: 변경사항을 잠시 임배류하고 싶어요. 아직 커밋은 안 만들래요
6. **rebase**: merge와 달리, 두 개의 브랜치를 병합하는 과정에서 커밋라인을 하나로 만들어 히스토리를 깔끔하게 볼 수 있게 함
