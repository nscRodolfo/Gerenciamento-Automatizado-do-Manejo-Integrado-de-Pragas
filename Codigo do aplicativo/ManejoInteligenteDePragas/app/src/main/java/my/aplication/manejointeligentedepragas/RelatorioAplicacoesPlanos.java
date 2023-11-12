package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Window;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;

import com.example.manejointeligentedepragas.R;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.helper.DateAsXAxisLabelFormatter;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.DataPointInterface;
import com.jjoe64.graphview.series.LineGraphSeries;
import com.jjoe64.graphview.series.OnDataPointTapListener;
import com.jjoe64.graphview.series.PointsGraphSeries;
import com.jjoe64.graphview.series.Series;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class RelatorioAplicacoesPlanos extends AppCompatActivity {

    int Cod_Propriedade;
    int codCultura;
    int codPraga;
    String nome;
    boolean aplicado;
    String nomePropriedade;
    String nomePraga;

    ArrayList<Integer> popPragas = new ArrayList<Integer>();
    ArrayList<Integer> numPlantas = new ArrayList<Integer>();
    ArrayList<String> metodos = new ArrayList<String>();
    ArrayList<Date> dataContagem = new ArrayList<Date>();
    ArrayList<Date> dataContagemPlano = new ArrayList<Date>();
    ArrayList<Long> dataContagemLong = new ArrayList<Long>();


    private Dialog mDialog;

    /*private XYPlot grafico;*/

    //data
    SimpleDateFormat formataData = new SimpleDateFormat("yyyy-MM-dd");

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_relatorio_aplicacoes_planos);
        openDialog();

        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        codPraga = getIntent().getIntExtra("Cod_Praga", 0);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        nomePraga = getIntent().getStringExtra("nomePraga");

        setTitle("MIP² | "+nome);

        GraphView graphA = (GraphView) findViewById(R.id.graphA);

        graphA.setTitle("Aplicações x População de pragas");
        graphA.setTitleColor(Color.parseColor("#659251"));

        resgataDados(codCultura,codPraga,graphA);
    }

    public void popularDadosGráfico(GraphView graph){
        int tamanho = dataContagem.size();
        DataPoint[] dataPoints = new DataPoint[tamanho];
        for (int i =0;i<tamanho;i++){
            dataPoints[i] = new DataPoint(dataContagem.get(i).getTime(),popPragas.get(i));
            //Toast.makeText(RelatorioAplicacoesPlanos.this, ""+dataPoints[i].toString(), Toast.LENGTH_LONG).show();
        }
        LineGraphSeries<DataPoint> series = new LineGraphSeries<DataPoint>(dataPoints);
        series.setColor(Color.parseColor("#659251"));
        PointsGraphSeries<DataPoint> series1 = new PointsGraphSeries<>(dataPoints);
        series1.setColor(Color.parseColor("#659251"));


        graph.getGridLabelRenderer().setLabelFormatter(new DateAsXAxisLabelFormatter(RelatorioAplicacoesPlanos.this));
        graph.getGridLabelRenderer().setNumHorizontalLabels(3);
        graph.getGridLabelRenderer().setHumanRounding(false,true);
        graph.getViewport().setMinX(dataContagem.get(0).getTime());
        graph.getViewport().setMaxX(dataContagem.get(tamanho-1).getTime());
        graph.getViewport().setXAxisBoundsManual(true);

        graph.getViewport().setScalable(true);
        graph.getViewport().setScalableY(true);
        graph.getViewport().setScrollable(true);
        graph.getViewport().setScrollableY(true);


        graph.addSeries(series);
        graph.addSeries(series1);

        mDialog.dismiss();

        series1.setOnDataPointTapListener(new OnDataPointTapListener() {
            @Override
            public void onTap(Series series, DataPointInterface dataPoint) {
                Double d = dataPoint.getX();
                Long aux = d.longValue();
                int indexInfos = dataContagemLong.indexOf(aux);

                String dataFormatadaAplicacao = formataData.format(dataContagem.get(indexInfos));
                String dataFormatadaPlano = formataData.format(dataContagemPlano.get(indexInfos));

                AlertDialog.Builder dlgBox = new AlertDialog.Builder(RelatorioAplicacoesPlanos.this);
                dlgBox.setTitle("Informações:");
                dlgBox.setMessage("\nMétodo aplicado: "+metodos.get(indexInfos)+"\n\nData da aplicação: "+dataFormatadaAplicacao+"\n\nNúmero de amostras: "+numPlantas.get(indexInfos)+"\n\nPlantas infestadas: "+popPragas.get(indexInfos) + "\n\nData da última contagem: "+dataFormatadaPlano);
                dlgBox.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                    }
                });

                dlgBox.show();
            }

        });

    }


    public void resgataDados(final int codCultura, final int codPraga, final GraphView graphA){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            mDialog.dismiss();
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/resgataDadosGraphAplicacao.php?Cod_Cultura="+codCultura+"&&Cod_Praga="+codPraga;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            try {
                                dataContagemPlano.add(formataData.parse(obj.getString("DataPlano")));
                                dataContagem.add(formataData.parse(obj.getString("DataAplicacao")));
                                dataContagemLong.add(formataData.parse(obj.getString("DataAplicacao")).getTime());
                            } catch (ParseException e) {
                                Toast.makeText(RelatorioAplicacoesPlanos.this, e.toString(), Toast.LENGTH_LONG).show();
                            }
                            popPragas.add(obj.getInt("popPragas"));
                            numPlantas.add(obj.getInt("numPlantas"));
                            metodos.add(obj.getString("Metodo"));
                        }
                        if(dataContagem.size() == 0){
                            AlertDialog.Builder dlgBox = new AlertDialog.Builder(RelatorioAplicacoesPlanos.this);
                            dlgBox.setCancelable(false);
                            dlgBox.setTitle("Aviso!");
                            dlgBox.setMessage("Você não realizou nenhum plano de amostragem para esta praga, deseja realizar um agora?");
                            dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    Intent i = new Intent(RelatorioAplicacoesPlanos.this, PlanoDeAmostragem.class);
                                    i.putExtra("Cod_Propriedade", Cod_Propriedade);
                                    i.putExtra("Cod_Cultura", codCultura);
                                    i.putExtra("NomeCultura", nome);
                                    i.putExtra("nomePraga", nomePraga);
                                    i.putExtra("Cod_Praga", codPraga);
                                    i.putExtra("Aplicado", aplicado);
                                    i.putExtra("nomePropriedade", nomePropriedade);
                                    startActivity(i);
                                }
                            });
                            dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    Intent i = new Intent(RelatorioAplicacoesPlanos.this, AcoesCultura.class);
                                    i.putExtra("Cod_Cultura", codCultura);
                                    i.putExtra("NomeCultura", nome);
                                    i.putExtra("Cod_Propriedade", Cod_Propriedade);
                                    i.putExtra("Aplicado", aplicado);
                                    i.putExtra("nomePropriedade", nomePropriedade);
                                    startActivity(i);
                                }
                            });
                            dlgBox.show();
                        }else{
                            popularDadosGráfico(graphA);
                        }

                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(RelatorioAplicacoesPlanos.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(RelatorioAplicacoesPlanos.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();
        // mDialog.dismiss(); -> para fechar a dialog

    }
}
