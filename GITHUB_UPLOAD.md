# دستورالعمل آپلود پروژه در GitHub

## مرحله 1: ایجاد Repository در GitHub

1. به آدرس https://github.com/new بروید
2. نام repository را وارد کنید (مثلاً: `qlora-cnn-dailymail-summarization`)
3. Public یا Private را انتخاب کنید
4. **توجه:** گزینه‌های "Initialize with README" یا "Add .gitignore" را انتخاب نکنید
5. روی "Create repository" کلیک کنید

## مرحله 2: اتصال و Push

بعد از ایجاد repository، دستورات زیر را در terminal اجرا کنید:

```bash
# اضافه کردن remote (نام repository و username خود را جایگزین کنید)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

# تغییر نام branch به main (اگر لازم باشد)
git branch -M main

# Push کردن به GitHub
git push -u origin main
```

## مثال:

اگر username شما `john` و repository name شما `qlora-summarization` باشد:

```bash
git remote add origin https://github.com/john/qlora-summarization.git
git branch -M main
git push -u origin main
```

## اگر از SSH استفاده می‌کنید:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git branch -M main
git push -u origin main
```

## نکات مهم:

- اگر از HTTPS استفاده می‌کنید، GitHub از شما username و password (یا Personal Access Token) می‌خواهد
- برای امنیت بیشتر، از Personal Access Token استفاده کنید (Settings → Developer settings → Personal access tokens)
- اگر خطای authentication دریافت کردید، از Personal Access Token به جای password استفاده کنید

