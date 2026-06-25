import os
import time
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

# --- הגדרות קבצים ---
# קובץ הקלט הוא הקובץ שנוצר מההרצה הקודמת (המכיל את השורות המשוכפלות)
input_file = "stress_media_merged_with_duplicates_final2.xlsx"
# קובץ הפלט החדש והמתוקן שייווצר בסיום
output_file = "stress_media_merged_with_duplicates_final.xlsx"

column_to_translate = 5  # מיקום העמודה בעברית (עמודה שישית, אינדקס 5)
target_column_name = "title_en"  # שם העמודה של התרגום לאנגלית


def main():
    if not os.path.exists(input_file):
        print(f"שגיאה: לא מצאתי את הקובץ '{input_file}' בתיקייה!")
        print("ודאי שקובץ האקסל נמצא באותה תיקייה של הקוד ושמו מעודכן בשורה 9 בקוד.")
        return

    print(f"טוען את הקובץ '{input_file}' לצורך תיקון והשלמת שורות כפולות...")
    df = pd.read_excel(input_file)

    # ודאות שסוג הנתונים בעמודת התרגום הוא טקסט תקין
    if target_column_name not in df.columns:
        df[target_column_name] = ""

    # ניקוי רווחים קל כדי שההשוואה בין העמודות תהיה מדויקת ב-100%
    df[target_column_name] = df[target_column_name].astype(str).str.strip()

    # הגדרת המתרגם עם קוד השפה הנכון והנתמך ('iw' לעברית)
    translator = GoogleTranslator(source='iw', target='en')

    # 🎯 פילטר חכם: מוצאים רק שורות שבהן האנגלית שווה לעברית, או שהיא ריקה/NaN/שגיאה
    def needs_translation(row):
        val_he = str(row.iloc[column_to_translate]).strip()
        val_en = str(row[target_column_name]).strip()

        # אם התא ריק או מכיל שגיאה
        if pd.isna(row[target_column_name]) or val_en == "" or val_en == "nan" or val_en == "ERROR_TRANSLATING":
            return True
        # הקסם: אם הטקסט באנגלית נשאר זהה לחלוטין לעברית, הוא יסומן לתרגום מחדש!
        if val_he == val_en:
            return True
        return False

    # מסמנים ומסננים את השורות שבאמת דורשות טיפול
    mask = df.apply(needs_translation, axis=1)
    rows_to_translate = df[mask]

    total_missing = len(rows_to_translate)
    print(f"נמצאו {total_missing} שורות שבהן התרגום נכשל או חזר על עצמו בעברית.")

    if total_missing == 0:
        print("כל השורות מתורגמות ומפוקסות לאנגלית! אין מה לתקן.")
        return

    # ריצה על השורות הבעייתיות בלבד
    for idx, row in tqdm(rows_to_translate.iterrows(), total=total_missing, desc="מתקן שורות בעברית"):
        text_he = row.iloc[column_to_translate]

        if pd.isna(text_he) or str(text_he).strip() == "":
            df.at[idx, target_column_name] = ""
            continue

        try:
            # תרגום מחדש
            translated_text = translator.translate(str(text_he))
            df.at[idx, target_column_name] = translated_text

            # השהייה קלה כדי לא להעמיס על שרתי גוגל
            time.sleep(0.15)

        except Exception as e:
            # במקרה של שגיאה רגעית באינטרנט, נרשום שגיאה וננוח 2 שניות
            df.at[idx, target_column_name] = "ERROR_TRANSLATING"
            time.sleep(2)

            # שמירת גיבוי אוטומטית לקובץ הסופי כל 50 שורות שתורגמו
        if (idx + 1) % 50 == 0:
            df.to_excel(output_file, index=False)

    # שמירה סופית לאחר שכל השורות הסתיימו
    df.to_excel(output_file, index=False)
    print(f"\nהתיקון וההשלמה הסתיימו בהצלחה! הקובץ הסופי נשמר בשם: {output_file}")


if __name__ == "__main__":
    main()