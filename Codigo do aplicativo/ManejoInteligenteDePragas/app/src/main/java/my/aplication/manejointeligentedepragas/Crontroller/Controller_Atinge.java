package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.AtingeModel;

public class Controller_Atinge {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Atinge(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addAtinge(AtingeModel atingeModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Atinge, atingeModel.getCod_Atinge());
        cv.put(bancoLocal.NivelDeControle, atingeModel.getNivelDeControle());
        cv.put(bancoLocal.NumeroPlantasAmostradas, atingeModel.getNumeroPlantasAmostradas());
        cv.put(bancoLocal.PontosPorTalhao, atingeModel.getPontosPorTalhao());
        cv.put(bancoLocal.PlantasPorPonto, atingeModel.getPlantasPorPonto());
        cv.put(bancoLocal.fk_Praga_Cod_Praga, atingeModel.getFk_Cod_Praga());
        cv.put(bancoLocal.fk_Planta_Cod_Planta, atingeModel.getFk_Cod_Planta());
        cv.put(bancoLocal.NumAmostras, atingeModel.getNumAmostras());
        return db.insert(bancoLocal.Atinge, null, cv) != -1;
    }

    public boolean updateAtinge(AtingeModel atingeModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Atinge, atingeModel.getCod_Atinge());
        cv.put(bancoLocal.NivelDeControle, atingeModel.getNivelDeControle());
        cv.put(bancoLocal.NumeroPlantasAmostradas, atingeModel.getNumeroPlantasAmostradas());
        cv.put(bancoLocal.PontosPorTalhao, atingeModel.getPontosPorTalhao());
        cv.put(bancoLocal.PlantasPorPonto, atingeModel.getPlantasPorPonto());
        cv.put(bancoLocal.fk_Praga_Cod_Praga, atingeModel.getFk_Cod_Praga());
        cv.put(bancoLocal.fk_Planta_Cod_Planta, atingeModel.getFk_Cod_Planta());
        cv.put(bancoLocal.NumAmostras, atingeModel.getNumAmostras());
        String where = " Cod_Atinge = '"+ atingeModel.getCod_Atinge()+"'";
        return db.update(bancoLocal.Atinge, cv, where, null) > 0;
    }

    public AtingeModel getAtinge(int codPraga, int codPlanta) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Atinge+" WHERE fk_Praga_Cod_Praga="+codPraga+" AND fk_Planta_Cod_Planta="+codPlanta;
        AtingeModel atingeModel = new AtingeModel();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                atingeModel.setCod_Atinge(c.getInt(0));
                atingeModel.setNivelDeControle(c.getFloat(1));
                atingeModel.setNumeroPlantasAmostradas(c.getInt(2));
                atingeModel.setPontosPorTalhao(c.getInt(3));
                atingeModel.setPlantasPorPonto(c.getInt(4));
                atingeModel.setFk_Cod_Praga(c.getInt(5));
                atingeModel.setFk_Cod_Planta(c.getInt(6));
                atingeModel.setNumAmostras(c.getInt(7));
            } while (c.moveToNext());
        }
        c.close();
        return atingeModel;
    }

    public void removerAtinge(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Atinge, query, null);
    }


}
